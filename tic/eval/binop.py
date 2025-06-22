import ast
import operator
from typing import Any

from .base import BaseEvaluator
from .error import EvalError

# Mapping from ast operator nodes to Python's operator functions
OPERATOR_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
}

COMPARISON_MAP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

UNARY_OPERATOR_MAP = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
    ast.Not: operator.not_,
    ast.Invert: operator.inv,
}


class BinOpEvaluator(BaseEvaluator):
    """A mixin for evaluating binary operation nodes."""

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        """Handles binary operations like +, -, *, /."""
        left_val = self.visit(node.left)
        right_val = self.visit(node.right)
        op_func = OPERATOR_MAP.get(type(node.op))
        if not op_func:
            raise EvalError(f"Operator {type(node.op).__name__} not supported.", node)
        try:
            return op_func(left_val, right_val)
        except Exception as e:
            raise EvalError(f"Failed to execute operation: {e}", node, cause=e)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        """Handles unary operations like -, not, ~."""
        operand_val = self.visit(node.operand)
        op_func = UNARY_OPERATOR_MAP.get(type(node.op))
        if not op_func:
            raise EvalError(
                f"Unary operator {type(node.op).__name__} not supported.", node
            )
        try:
            return op_func(operand_val)
        except Exception as e:
            raise EvalError(f"Failed to execute unary operation: {e}", node, cause=e)

    def visit_Compare(self, node: ast.Compare) -> bool:
        """Handles comparison operations."""
        left_val = self.visit(node.left)
        # TODO: This doesn't support chained comparisons like `1 < x < 10`.
        if len(node.ops) != 1:
            raise EvalError("Chained comparisons are not yet supported.", node)

        op_node = node.ops[0]
        op_func = COMPARISON_MAP.get(type(op_node))
        if not op_func:
            raise EvalError(
                f"Comparison operator {type(op_node).__name__} not supported.", node
            )

        right_val = self.visit(node.comparators[0])
        try:
            return op_func(left_val, right_val)
        except Exception as e:
            raise EvalError(f"Failed to execute comparison: {e}", node, cause=e)
