import ast
from dataclasses import dataclass
from typing import Any

from ..state import State
from ..state.closure import LiveClosureState
from ..state.scoped import Scoped
from .analysis import get_free_variables
from .base import BaseEvaluator


class _ReturnException(Exception):
    """Internal exception to signal a return statement, carrying the return value."""

    def __init__(self, value: Any):
        self.value = value


@dataclass
class UserFunction:
    """Represents a user-defined function and its closure."""

    name: str
    args: ast.arguments
    body: list[ast.stmt]
    closure_state: State  # A *reference* to the state where the function was defined.
    source_text: str | None = None

    def __call__(self, *args, **kwargs) -> Any:
        # Defer to the new execute method, providing a way for Python code
        # to call this function.
        return self.execute(list(args), kwargs, self.source_text)

    def execute(self, args: list, kwargs: dict, full_source_code: str | None) -> Any:
        """Execute the function with a new evaluator."""
        # This needs to be imported here to avoid a circular dependency
        from tic.eval.arguments import bind_arguments
        from tic.eval.core import Evaluator

        # The evaluator needs the source to provide good error messages
        evaluator = Evaluator(source_code=full_source_code)

        # Step 1: Create the execution scope. The Scoped state will create its
        # own Ephemeral local store for arguments.
        exec_state = Scoped(self.closure_state)

        # Step 2: Bind arguments to the new local scope. Note that we are just
        # passing the raw Python values, not visiting them. This is because
        # we are at the boundary between Python and the tic interpreter.
        bound_args = bind_arguments(self.name, self.args, args, kwargs)
        for name, value in bound_args.items():
            exec_state.set(name, value)

        # Step 3: Execute the function's body within the new scope.
        evaluator.state = exec_state
        try:
            for node in self.body:
                evaluator.visit(node)
            return None  # No explicit return means the function returns None
        except _ReturnException as e:
            return e.value


class FunctionEvaluator(BaseEvaluator):
    """A mixin for evaluating function definition and return nodes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles function definitions."""
        free_vars = get_free_variables(node)
        closure = LiveClosureState(self.state, free_vars)

        func = UserFunction(
            name=node.name,
            args=node.args,
            body=node.body,
            closure_state=closure,
            source_text=ast.get_source_segment(self.source_code, node),
        )
        self.state.set(node.name, func)

    def visit_Lambda(self, node: ast.Lambda) -> UserFunction:
        """Handles lambda expressions."""
        free_vars = get_free_variables(node)
        closure = LiveClosureState(self.state, free_vars)

        return UserFunction(
            name="<lambda>",
            args=node.args,
            body=[ast.Return(value=node.body)],  # Lambdas are a single expression
            closure_state=closure,
            source_text=ast.get_source_segment(self.source_code, node),
        )

    def visit_Return(self, node: ast.Return) -> None:
        """Handles return statements."""
        value = self.visit(node.value) if node.value else None
        raise _ReturnException(value)
