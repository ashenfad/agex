import ast
import inspect
from typing import Any

from ..agent.datatypes import TaskSuccess, _AgentExit
from .base import BaseEvaluator
from .builtins import STATEFUL_BUILTINS, _print_stateful
from .error import EvalError
from .functions import UserFunction
from .objects import AgexClass, AgexDataClass
from .user_errors import (
    AgexError,
    AgexIndexError,
    AgexKeyError,
    AgexTypeError,
    AgexValueError,
)
from .validation import validate_with_sampling


class CallEvaluator(BaseEvaluator):
    """A mixin for evaluating function call nodes."""

    def _unwrap_bound_objects(
        self, args: list[Any], kwargs: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Unwrap BoundInstanceObject arguments for external function calls.

        When agents pass registered live objects to external libraries (like pandas),
        the libraries expect the actual underlying object, not our wrapper.
        """
        from .objects import BoundInstanceObject

        # Unwrap args
        unwrapped_args = []
        for arg in args:
            if isinstance(arg, BoundInstanceObject):
                # Get the actual live object from the host registry
                live_instance = arg.host_registry[arg.reg_object.name]
                unwrapped_args.append(live_instance)
            else:
                unwrapped_args.append(arg)

        # Unwrap kwargs
        unwrapped_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, BoundInstanceObject):
                # Get the actual live object from the host registry
                live_instance = value.host_registry[value.reg_object.name]
                unwrapped_kwargs[key] = live_instance
            else:
                unwrapped_kwargs[key] = value

        return unwrapped_args, unwrapped_kwargs

    def _handle_secure_format(
        self,
        format_str: str,
        args_nodes: list[ast.expr],
        kwargs_nodes: list[ast.keyword],
    ) -> str:
        """
        Secure format string handling that prevents sandbox escapes.

        This method evaluates format arguments through the sandbox (so attribute access
        is properly validated), then uses a custom formatter to prevent bypass attacks.
        """
        from string import Formatter

        # Evaluate all arguments through the sandbox first
        args = [self.visit(arg) for arg in args_nodes]
        kwargs = {kw.arg: self.visit(kw.value) for kw in kwargs_nodes if kw.arg}

        class SandboxFormatter(Formatter):
            def __init__(self, evaluator):
                self.evaluator = evaluator
                super().__init__()

            def get_field(self, field_name, args, kwargs):
                # Parse field like "obj.attr" or "0.attr"
                parts = field_name.split(".")
                if len(parts) == 1:
                    # Simple field like {name} or {0} - allow these
                    return super().get_field(field_name, args, kwargs)

                # Complex field with attribute access - this is what we need to block
                # Since the arguments were already evaluated through our sandbox,
                # we know the base objects are safe. But we need to prevent
                # the format string from accessing additional attributes.

                # For now, we'll be conservative and block any dotted field access
                # Users should use f-strings for complex expressions
                raise AgexError(
                    f"Format string attribute access '{field_name}' is not allowed. Use f-strings instead."
                )

        formatter = SandboxFormatter(self)
        try:
            return formatter.format(format_str, *args, **kwargs)
        except Exception as e:
            # Re-raise with better context
            raise EvalError(f"Format string error: {e}", None) from e

    def visit_Call(self, node: ast.Call) -> Any:
        """Handles function calls."""

        # Special handling for string.format() calls to prevent sandbox escapes
        if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            # Check if this is a string literal .format() call
            if isinstance(node.func.value, ast.Constant) and isinstance(
                node.func.value.value, str
            ):
                return self._handle_secure_format(
                    node.func.value.value, node.args, node.keywords
                )

            # Check if this is a string variable .format() call
            string_obj = self.visit(node.func.value)
            if isinstance(string_obj, str):
                return self._handle_secure_format(string_obj, node.args, node.keywords)

        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords if kw.arg}

        # Handle stateful builtins first, as they need dependency injection.
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            if (stateful_fn_wrapper := STATEFUL_BUILTINS.get(fn_name)) is not None:
                try:
                    # Special cases for functions that need state but not evaluator
                    if fn_name == "print":
                        return _print_stateful(
                            *args,
                            state=self.state,
                            agent_name=self.agent.name,
                            on_event=self.on_event,
                        )
                    elif fn_name == "view_image":
                        from .builtins import _view_image_stateful

                        return _view_image_stateful(
                            *args,
                            **kwargs,
                            state=self.state,
                            agent_name=self.agent.name,
                            on_event=self.on_event,
                        )
                    elif fn_name == "task_continue":
                        from .builtins import _task_continue_with_observations

                        return _task_continue_with_observations(
                            *args,
                            state=self.state,
                            agent_name=self.agent.name,
                            on_event=self.on_event,
                        )

                    if stateful_fn_wrapper.needs_evaluator:
                        return stateful_fn_wrapper.fn(self, *args, **kwargs)
                    else:
                        # For builtins that don't need the evaluator
                        return stateful_fn_wrapper.fn(*args, **kwargs)
                except AgexError:
                    raise
                except Exception as e:
                    if isinstance(e, _AgentExit):
                        raise e
                    raise EvalError(
                        f"Error calling stateful builtin function '{fn_name}': {e}",
                        node,
                        cause=e,
                    )

        fn = self.visit(node.func)

        try:
            # Handle calling a AgexClass to create an instance
            if isinstance(fn, (AgexClass, AgexDataClass)):
                return fn(*args, **kwargs)

            # Legacy direct UserFunction execution path (explicit to handle signature)
            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code, parent_evaluator=self)

            # If this is a dual-decorated function needing state injection, route via proxy
            if hasattr(fn, "__agex_task_namespace__"):
                from .functions import TaskProxy

                proxy = TaskProxy(self, getattr(fn, "fn", fn))
                return proxy.execute(args, kwargs)

            # If function has a unified execute() hook, use it
            if hasattr(fn, "execute") and callable(getattr(fn, "execute")):
                return fn.execute(args, kwargs)  # type: ignore[attr-defined]

            if not callable(fn):
                fn_name_for_error = getattr(
                    node.func, "attr", getattr(node.func, "id", "object")
                )
                raise AgexError(f"'{fn_name_for_error}' is not callable.", node)

                # Check if this is a dual-decorated function needing state injection
            else:
                # Regular function call - no timer changes needed
                # For external functions, unwrap BoundInstanceObject arguments
                unwrapped_args, unwrapped_kwargs = self._unwrap_bound_objects(
                    args, kwargs
                )
                result = fn(*unwrapped_args, **unwrapped_kwargs)

            # Special handling for agent exit signals
            if isinstance(result, _AgentExit):
                # If this is a TaskSuccess signal, validate the result first
                if isinstance(result, TaskSuccess):
                    return_type = self.state.get("__expected_return_type__")
                    # Only validate if there's a return type and it's not inspect._empty
                    if return_type and return_type is not inspect.Parameter.empty:
                        try:
                            # The 'result' here is the TaskSuccess instance.
                            # We need to validate the value it's carrying.
                            validate_with_sampling(result.result, return_type)
                        except Exception as e:
                            # Re-raise as a AgexError to be caught by the loop
                            # Use clean type names for all types when possible
                            if (
                                hasattr(return_type, "__module__")
                                and hasattr(return_type, "__name__")
                                and not hasattr(
                                    return_type, "__origin__"
                                )  # Not a generic type
                            ):
                                # Use the clean class name for simple types (str, int, custom classes)
                                type_name = return_type.__name__
                            else:
                                # For generic types (list[int], dict[str, int]) or complex types,
                                # use the full string representation to preserve type parameters
                                type_name = str(return_type)
                            raise AgexError(
                                f"Output validation failed. The returned value did not match the expected type '{type_name}'.\nDetails: {e}",
                                node,
                            ) from e
                raise result

            return result
        except AgexError:
            # Re-raise user-facing errors directly without wrapping
            raise
        except ValueError as e:
            # Map ValueError to AgexValueError so agents can catch it
            raise AgexValueError(str(e), node) from e
        except TypeError as e:
            # Map TypeError to AgexTypeError so agents can catch it
            raise AgexTypeError(str(e), node) from e
        except KeyError as e:
            # Map KeyError to AgexKeyError so agents can catch it
            raise AgexKeyError(str(e), node) from e
        except IndexError as e:
            # Map IndexError to AgexIndexError so agents can catch it
            raise AgexIndexError(str(e), node) from e
        except Exception as e:
            # Check for registered exception mappings from live objects
            from .objects import BoundInstanceMethod

            if isinstance(fn, BoundInstanceMethod):
                # Check the registered object's exception mappings
                for exc_type, agex_exc_type in fn.reg_object.exception_mappings.items():
                    if isinstance(e, exc_type):
                        raise agex_exc_type(str(e), node) from e

            if isinstance(e, _AgentExit):
                raise e
            fn_name_for_error = getattr(
                node.func, "attr", getattr(node.func, "id", "object")
            )
            raise EvalError(f"Error calling '{fn_name_for_error}': {e}", node, cause=e)
