import ast

from .base import BaseEvaluator
from .user_errors import AgexValueError


class _BreakException(Exception):
    """Internal exception to signal a break statement."""


class _ContinueException(Exception):
    """Internal exception to signal a continue statement."""


def _safe_bool_eval(
    value, node: ast.AST, context_description: str = "Boolean evaluation"
) -> bool:
    """
    Safely evaluate a value in a boolean context, catching pandas and other exceptions
    that occur when trying to evaluate ambiguous values as booleans.

    Args:
        value: The value to evaluate in boolean context
        node: The AST node for error reporting
        context_description: Description of the context for error messages

    Returns:
        The boolean result

    Raises:
        AgexValueError: If the value cannot be evaluated as a boolean with line/col info
    """
    try:
        return bool(value)
    except ValueError as e:
        # Handle pandas "truth value ambiguous" and similar errors
        raise AgexValueError(str(e), node) from e
    except Exception as e:
        # Handle other boolean conversion errors
        raise AgexValueError(f"{context_description} error: {e}", node) from e


class LoopEvaluator(BaseEvaluator):
    """A mixin for evaluating loops and other control flow nodes."""

    def visit_If(self, node: ast.If) -> None:
        """Handles if, elif, and else statements."""
        test_result = self.visit(node.test)
        if _safe_bool_eval(test_result, node.test, "If statement condition"):
            for sub_node in node.body:
                self.visit(sub_node)
        else:
            for sub_node in node.orelse:
                self.visit(sub_node)

    def visit_Break(self, node: ast.Break) -> None:
        """Handles break statements."""
        raise _BreakException()

    def visit_Continue(self, node: ast.Continue) -> None:
        """Handles continue statements."""
        raise _ContinueException()

    def visit_For(self, node: ast.For) -> None:
        """Handles for loops."""
        iterable = self.visit(node.iter)
        did_break = False
        for item in iterable:
            try:
                self._handle_destructuring_assignment(node.target, item)
                for sub_node in node.body:
                    self.visit(sub_node)
            except _ContinueException:
                continue
            except _BreakException:
                did_break = True
                break

        if not did_break:
            for sub_node in node.orelse:
                self.visit(sub_node)

    def visit_While(self, node: ast.While) -> None:
        """Handles while loops."""
        did_break = False
        while True:
            test_result = self.visit(node.test)
            if not _safe_bool_eval(test_result, node.test, "While loop condition"):
                break

            try:
                for sub_node in node.body:
                    self.visit(sub_node)
            except _ContinueException:
                continue
            except _BreakException:
                did_break = True
                break

        if not did_break:
            for sub_node in node.orelse:
                self.visit(sub_node)
