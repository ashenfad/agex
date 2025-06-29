import ast
from typing import Any

from ..agent.datatypes import _AgentExit
from ..state import Namespaced
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


class CallEvaluator(BaseEvaluator):
    """A mixin for evaluating function call nodes."""

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
                    # Special case for print, which doesn't get evaluator but needs state
                    if fn_name == "print":
                        return _print_stateful(*args, state=self.state)

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

            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code)

            if not callable(fn):
                fn_name_for_error = getattr(
                    node.func, "attr", getattr(node.func, "id", "object")
                )
                raise AgexError(f"'{fn_name_for_error}' is not callable.", node)

            # Check if this is a dual-decorated function needing state injection
            if hasattr(fn, "__agex_task_namespace__"):
                # Create namespaced state for sub-agent
                namespace = fn.__agex_task_namespace__
                namespaced_state = Namespaced(self.state, namespace)
                kwargs["state"] = namespaced_state

            result = fn(*args, **kwargs)

            # Special handling for agent exit signals
            if isinstance(fn, type) and issubclass(fn, _AgentExit):
                raise result  # type: ignore

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
            if isinstance(e, _AgentExit):
                raise e
            fn_name_for_error = getattr(
                node.func, "attr", getattr(node.func, "id", "object")
            )
            raise EvalError(f"Error calling '{fn_name_for_error}': {e}", node, cause=e)
