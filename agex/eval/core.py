import ast
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.state.core import State

from .base import BaseEvaluator
from .binop import BinOpEvaluator
from .call import CallEvaluator
from .comprehension import ComprehensionEvaluator
from .error import EvalError
from .expressions import ExpressionEvaluator
from .functions import FunctionEvaluator, _ReturnException
from .loops import LoopEvaluator
from .resolver import Resolver
from .statements import StatementEvaluator


class Evaluator(
    CallEvaluator,
    BinOpEvaluator,
    ExpressionEvaluator,
    ComprehensionEvaluator,
    LoopEvaluator,
    FunctionEvaluator,
    StatementEvaluator,
    BaseEvaluator,
):
    """
    The main evaluator, composed of modular mixins from other files.
    """

    def __init__(
        self,
        agent: BaseAgent,
        state: State,
        source_code: str | None = None,
        timeout_seconds: float | None = None,
        start_time: float | None = None,
        sub_agent_time: float = 0.0,
        on_event: Callable[[Any], None] | None = None,
    ):
        actual_timeout = (
            timeout_seconds if timeout_seconds is not None else agent.timeout_seconds
        )
        super().__init__(
            agent,
            state,
            actual_timeout,
            start_time=start_time,
            sub_agent_time=sub_agent_time,
        )
        self.source_code = source_code
        self.resolver = Resolver(agent)
        self.on_event = on_event

    def visit_Module(self, node: ast.Module):
        """Evaluates a module by visiting each statement in its body."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr):
        """
        Handles expressions that are used as statements.
        The result of the expression is calculated but not stored.
        """
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            # avoid printing for builtins that already print
            if node.value.func.id in ("print", "view_image", "dir", "help"):
                # Still need to visit the call to execute it for its side effect.
                self.visit(node.value)
                return

        result = self.visit(node.value)

        if result is not None:
            # we auto-print here to mimic notebook behavior
            from .builtins import _print_stateful

            _print_stateful(
                result,
                state=self.state,
                agent_name=self.agent.name,
                on_event=self.on_event,
            )


def evaluate_program(
    program: str,
    agent: BaseAgent,
    state: State,
    timeout_seconds: float | None = None,
    on_event: Callable[[Any], None] | None = None,
):
    """
    Updates state with the result of running the program. The agent provides
    whitelisted functions and classes valid for the program.

    Args:
        program: The Python code to execute
        agent: The agent providing the execution context
        state: The state to execute in
        timeout_seconds: Optional timeout override. If None, uses agent.timeout_seconds
        on_event: Optional handler to call for each event
    """
    actual_timeout = (
        timeout_seconds if timeout_seconds is not None else agent.timeout_seconds
    )
    tree = ast.parse(program)
    evaluator = Evaluator(
        agent,
        state,
        source_code=program,
        timeout_seconds=actual_timeout,
        on_event=on_event,
    )

    try:
        evaluator.visit(tree)
    except _ReturnException as e:
        # Convert return statement outside function to a helpful error
        raise EvalError(
            "'return' outside function. You're in an agent environment, not a regular Python function. "
            "Use task_success(result) to complete your task, not return.",
            e.node,
        )
