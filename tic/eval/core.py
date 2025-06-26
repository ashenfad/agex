import ast

from tic.agent.base import BaseAgent
from tic.state.core import State

from .base import BaseEvaluator
from .binop import BinOpEvaluator
from .call import CallEvaluator
from .comprehension import ComprehensionEvaluator
from .expressions import ExpressionEvaluator
from .functions import FunctionEvaluator
from .loops import LoopEvaluator
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
    ):
        actual_timeout = (
            timeout_seconds if timeout_seconds is not None else agent.timeout_seconds
        )
        super().__init__(agent, state, actual_timeout)
        self.source_code = source_code

    def visit_Module(self, node: ast.Module):
        """Evaluates a module by visiting each statement in its body."""
        for stmt in node.body:
            self.visit(stmt)

    def visit_Expr(self, node: ast.Expr):
        """
        Handles expressions that are used as statements.
        The result of the expression is calculated but not stored.
        """
        self.visit(node.value)


def evaluate_program(
    program: str, agent: BaseAgent, state: State, timeout_seconds: float | None = None
):
    """
    Updates state with the result of running the program. The agent provides
    whitelisted functions and classes valid for the program.

    Args:
        program: The Python code to execute
        agent: The agent providing the execution context
        state: The state to execute in
        timeout_seconds: Optional timeout override. If None, uses agent.timeout_seconds
    """
    actual_timeout = (
        timeout_seconds if timeout_seconds is not None else agent.timeout_seconds
    )
    tree = ast.parse(program)
    evaluator = Evaluator(
        agent, state, source_code=program, timeout_seconds=actual_timeout
    )
    evaluator.visit(tree)
