import ast
from dataclasses import dataclass
from typing import Any

from tic.agent import Agent

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
    agent: Agent | None = None  # The agent context this function was defined in.

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.agent:
            raise RuntimeError(
                "UserFunction cannot be called directly without an Agent context."
            )
        # The agent provides the top-level source code for consistent errors.
        source_code = getattr(self.agent, "_source_code_for_eval", None)
        return self.execute(list(args), kwargs, source_code)

    def execute(self, args: list, kwargs: dict, source_code: str | None):
        """Execute the function with a new evaluator."""
        from tic.eval.arguments import bind_arguments
        from tic.eval.core import Evaluator

        exec_state = Scoped(self.closure_state)

        bound_args = bind_arguments(self.name, self.args, args, kwargs)
        for name, value in bound_args.items():
            exec_state.set(name, value)

        if not self.agent:
            raise RuntimeError("Cannot execute function without an agent context.")

        evaluator = Evaluator(
            agent=self.agent,
            state=exec_state,
            source_code=source_code,
        )
        try:
            for node in self.body:
                evaluator.visit(node)
            return None
        except _ReturnException as e:
            return e.value


class FunctionEvaluator(BaseEvaluator):
    """A mixin for evaluating function definition and return nodes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles function definitions."""
        free_vars = get_free_variables(node)
        closure = LiveClosureState(self.state, free_vars)

        source_text = None
        if self.source_code:
            source_text = ast.get_source_segment(self.source_code, node)

        func = UserFunction(
            name=node.name,
            args=node.args,
            body=node.body,
            closure_state=closure,
            source_text=source_text,
            agent=self.agent,
        )
        self.state.set(node.name, func)

    def visit_Lambda(self, node: ast.Lambda) -> UserFunction:
        """Handles lambda expressions."""
        free_vars = get_free_variables(node)
        closure = LiveClosureState(self.state, free_vars)

        source_text = None
        if self.source_code:
            source_text = ast.get_source_segment(self.source_code, node)

        return UserFunction(
            name="<lambda>",
            args=node.args,
            body=[ast.Return(value=node.body)],  # Lambdas are a single expression
            closure_state=closure,
            source_text=source_text,
            agent=self.agent,
        )

    def visit_Return(self, node: ast.Return) -> None:
        """Handles return statements."""
        value = self.visit(node.value) if node.value else None
        raise _ReturnException(value)
