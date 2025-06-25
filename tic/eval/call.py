import ast
from typing import Any

from ..agent import _AgentExit
from .base import BaseEvaluator
from .builtins import STATEFUL_BUILTINS, _print_stateful
from .error import EvalError
from .functions import UserFunction
from .objects import TicClass, TicDataClass
from .user_errors import (
    TicError,
    TicIndexError,
    TicKeyError,
    TicTypeError,
    TicValueError,
)


class CallEvaluator(BaseEvaluator):
    """A mixin for evaluating function call nodes."""

    def visit_Call(self, node: ast.Call) -> Any:
        """Handles function calls."""
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
                except TicError:
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
            # Handle calling a TicClass to create an instance
            if isinstance(fn, (TicClass, TicDataClass)):
                return fn(*args, **kwargs)

            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code)

            if not callable(fn):
                fn_name_for_error = getattr(
                    node.func, "attr", getattr(node.func, "id", "object")
                )
                raise TicError(f"'{fn_name_for_error}' is not callable.", node)

            result = fn(*args, **kwargs)

            # Special handling for agent exit signals
            if isinstance(fn, type) and issubclass(fn, _AgentExit):
                raise result  # type: ignore

            return result
        except TicError:
            # Re-raise user-facing errors directly without wrapping
            raise
        except ValueError as e:
            # Map ValueError to TicValueError so agents can catch it
            raise TicValueError(str(e), node) from e
        except TypeError as e:
            # Map TypeError to TicTypeError so agents can catch it
            raise TicTypeError(str(e), node) from e
        except KeyError as e:
            # Map KeyError to TicKeyError so agents can catch it
            raise TicKeyError(str(e), node) from e
        except IndexError as e:
            # Map IndexError to TicIndexError so agents can catch it
            raise TicIndexError(str(e), node) from e
        except Exception as e:
            if isinstance(e, _AgentExit):
                raise e
            fn_name_for_error = getattr(
                node.func, "attr", getattr(node.func, "id", "object")
            )
            raise EvalError(f"Error calling '{fn_name_for_error}': {e}", node, cause=e)
