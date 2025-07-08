import ast
import operator
from typing import Any

from .base import BaseEvaluator
from .error import EvalError
from .loops import _safe_bool_eval
from .user_errors import (
    AgexArithmeticError,
    AgexOverflowError,
    AgexTypeError,
    AgexZeroDivisionError,
)

# Mapping from ast operator nodes to Python's operator functions
OPERATOR_MAP = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
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
        except ZeroDivisionError as e:
            # Agent-catchable arithmetic error
            raise AgexZeroDivisionError(str(e), node) from e
        except OverflowError as e:
            # Agent-catchable arithmetic error
            raise AgexOverflowError(str(e), node) from e
        except (ArithmeticError, ValueError) as e:
            # Other arithmetic errors that agents should be able to catch
            raise AgexArithmeticError(str(e), node) from e
        except TypeError as e:
            # Type errors are also agent-catchable
            raise AgexTypeError(str(e), node) from e
        except Exception as e:
            # System-level errors remain as EvalError
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
        except ZeroDivisionError as e:
            # Agent-catchable arithmetic error
            raise AgexZeroDivisionError(str(e), node) from e
        except OverflowError as e:
            # Agent-catchable arithmetic error
            raise AgexOverflowError(str(e), node) from e
        except (ArithmeticError, ValueError) as e:
            # Other arithmetic errors that agents should be able to catch
            raise AgexArithmeticError(str(e), node) from e
        except TypeError as e:
            # Type errors are also agent-catchable
            raise AgexTypeError(str(e), node) from e
        except Exception as e:
            # System-level errors remain as EvalError
            raise EvalError(f"Failed to execute unary operation: {e}", node, cause=e)

    def visit_Compare(self, node: ast.Compare) -> bool:
        """Handles comparison operations."""
        left_val = self.visit(node.left)

        for op, comparator_node in zip(node.ops, node.comparators):
            right_val = self.visit(comparator_node)

            op_func = COMPARISON_MAP.get(type(op))
            if not op_func:
                raise EvalError(
                    f"Comparison operator {type(op).__name__} not supported.", node
                )

            try:
                result = op_func(left_val, right_val)

                # Check if result is a pandas-like object that shouldn't be boolean-evaluated
                # In such cases, we can't short-circuit and should just return the result
                if hasattr(result, "dtype") and hasattr(result, "__len__"):
                    # This looks like a pandas Series or numpy array
                    # For chained comparisons with pandas, we can't short-circuit
                    # so just continue with the next comparison
                    if len(node.ops) > 1:
                        # Multiple comparisons with pandas Series - this is complex
                        # For now, just return the result and let the user handle it
                        return result
                    else:
                        # Single comparison returning a Series - that's normal
                        return result
                else:
                    # Regular scalar comparison - use safe boolean evaluation for short-circuiting
                    if not _safe_bool_eval(
                        result, node, f"Comparison operation '{type(op).__name__}'"
                    ):
                        # Short-circuit
                        return False
            except TypeError as e:
                # Re-raise as a user-catchable error
                raise AgexTypeError(str(e), node) from e

            # The right value becomes the left value for the next comparison
            left_val = right_val

        return True
