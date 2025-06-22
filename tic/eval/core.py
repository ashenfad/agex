import ast

from tic.agent import Agent
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
        agent: Agent | None = None,
        state: State | None = None,
        source_code: str | None = None,
    ):
        super().__init__(agent, state)
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


def evaluate_program(program: str, agent: Agent, state: State):
    """
    Updates state with the result of running the program. The agent provides
    whitelisted functions and classes valid for the program.
    """
    tree = ast.parse(program)
    evaluator = Evaluator(agent, state, source_code=program)
    evaluator.visit(tree)
