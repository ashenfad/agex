import ast
from dataclasses import dataclass
from typing import Any, Callable

from ..agent import _AgentExit
from ..state import State
from .base import BaseEvaluator
from .builtins import _dir, _hasattr, _help
from .error import EvalError
from .functions import UserFunction
from .objects import PrintTuple, TicClass


@dataclass
class StatefulFn:
    """A wrapper for stateful builtins to declare their dependencies."""

    fn: Callable[..., Any]
    needs_evaluator: bool = False


def _print_stateful(*args: Any, state: State):
    """
    A custom implementation of 'print' that appends its arguments to the
    `__stdout__` list in the agent's state as a single `PrintTuple`.
    """
    # Ensure __stdout__ exists and is a list
    current_stdout = state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    # Append all arguments as a single entry
    new_stdout = current_stdout + [PrintTuple(args)]
    state.set("__stdout__", new_stdout)


STATEFUL_BUILTINS: dict[str, StatefulFn] = {
    "print": StatefulFn(_print_stateful, needs_evaluator=False),
    "help": StatefulFn(_help, needs_evaluator=True),
    "dir": StatefulFn(_dir, needs_evaluator=True),
    "hasattr": StatefulFn(_hasattr, needs_evaluator=True),
}


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
            if isinstance(fn, TicClass):
                return fn(*args, **kwargs)

            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code)

            if not callable(fn):
                fn_name_for_error = getattr(
                    node.func, "attr", getattr(node.func, "id", "object")
                )
                raise EvalError(f"'{fn_name_for_error}' is not callable.", node)

            result = fn(*args, **kwargs)

            # Special handling for agent exit signals
            if isinstance(fn, type) and issubclass(fn, _AgentExit):
                raise result  # type: ignore

            return result
        except Exception as e:
            if isinstance(e, _AgentExit):
                raise e
            fn_name_for_error = getattr(
                node.func, "attr", getattr(node.func, "id", "object")
            )
            raise EvalError(f"Error calling '{fn_name_for_error}': {e}", node, cause=e)
