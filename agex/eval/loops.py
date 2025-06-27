import ast

from .base import BaseEvaluator


class _BreakException(Exception):
    """Internal exception to signal a break statement."""


class _ContinueException(Exception):
    """Internal exception to signal a continue statement."""


class LoopEvaluator(BaseEvaluator):
    """A mixin for evaluating loops and other control flow nodes."""

    def visit_If(self, node: ast.If) -> None:
        """Handles if, elif, and else statements."""
        if self.visit(node.test):
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
        while self.visit(node.test):
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
