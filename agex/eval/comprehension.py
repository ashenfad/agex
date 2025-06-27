import ast
import contextlib
from typing import Generator

from .. import state
from .base import BaseEvaluator


class ComprehensionEvaluator(BaseEvaluator):
    """A mixin for evaluating comprehension nodes."""

    def _evaluate_comprehension_generators(
        self, generators: list[ast.comprehension]
    ) -> Generator[None, None, None]:
        """
        A recursive generator that handles the nested loops and 'if' conditions
        of a comprehension. It yields for each item that satisfies all conditions.
        """
        if not generators:
            yield
            return

        gen = generators[0]
        iterable = self.visit(gen.iter)

        for item in iterable:
            self._handle_destructuring_assignment(gen.target, item)

            all_ifs_passed = True
            for if_clause in gen.ifs:
                if not self.visit(if_clause):
                    all_ifs_passed = False
                    break
            if all_ifs_passed:
                yield from self._evaluate_comprehension_generators(generators[1:])

    @contextlib.contextmanager
    def _scoped_state(self):
        """A context manager to temporarily push a new scope onto the state."""
        original_state = self.state
        self.state = state.Scoped(original_state)
        try:
            yield
        finally:
            self.state = original_state

    def visit_ListComp(self, node: ast.ListComp) -> list:
        """Handles list comprehensions."""
        with self._scoped_state():
            result = []
            for _ in self._evaluate_comprehension_generators(node.generators):
                result.append(self.visit(node.elt))
            return result

    def visit_SetComp(self, node: ast.SetComp) -> set:
        """Handles set comprehensions."""
        with self._scoped_state():
            result = set()
            for _ in self._evaluate_comprehension_generators(node.generators):
                result.add(self.visit(node.elt))
            return result

    def visit_DictComp(self, node: ast.DictComp) -> dict:
        """Handles dictionary comprehensions."""
        with self._scoped_state():
            result = {}
            for _ in self._evaluate_comprehension_generators(node.generators):
                key = self.visit(node.key)
                value = self.visit(node.value)
                result[key] = value
            return result

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> list:
        """Handles generator expressions, materializing them into lists."""
        with self._scoped_state():
            result = []
            for _ in self._evaluate_comprehension_generators(node.generators):
                result.append(self.visit(node.elt))
            return result
