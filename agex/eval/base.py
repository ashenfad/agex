import ast
import time
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.state.core import State

from .error import EvalError
from .resolver import Resolver


class BaseEvaluator(ast.NodeVisitor):
    """A base class for evaluators, holding shared state."""

    def __init__(
        self,
        agent: "BaseAgent",
        state: "State",
        timeout_seconds: float = 5.0,
        start_time: float | None = None,
        sub_agent_time: float = 0.0,
    ):
        self.agent = agent
        self.state = state
        self.on_event: Callable[[Any], None] | None = None  # Will be set by Evaluator
        self.on_token: Callable[[Any], None] | None = None  # Will be set by Evaluator
        self.source_code: str | None = None
        self._start_time = start_time if start_time is not None else time.time()
        self._timeout_seconds = timeout_seconds
        self._sub_agent_time = sub_agent_time  # Total time spent in sub-agent calls
        self.resolver = Resolver(agent)  # Unified resolver for all lookups

    def _handle_destructuring_assignment(self, target_node: ast.AST, value: Any):
        """
        Recursively handles assignment to a name or a tuple.
        This is used for both standard assignment and comprehension targets.
        """
        if isinstance(target_node, ast.Starred):
            values = self._materialize_iterable_for_unpack(value, target_node)
            self._assign_sequence_targets([target_node], values, target_node)
            return

        if isinstance(target_node, (ast.Tuple, ast.List)):
            values = self._materialize_iterable_for_unpack(value, target_node)
            self._assign_sequence_targets(target_node.elts, values, target_node)
            return

        if isinstance(target_node, ast.Name):
            self.state.set(target_node.id, value)
        else:
            raise EvalError(
                "Assignment target must be a name, list, or tuple.", target_node
            )

    def _materialize_iterable_for_unpack(self, value: Any, node: ast.AST) -> list[Any]:
        """Convert an iterable into a concrete list for destructuring."""
        if not hasattr(value, "__iter__"):
            raise EvalError("Cannot unpack non-iterable value for assignment.", node)
        try:
            return list(value)
        except TypeError:
            raise EvalError("Cannot unpack non-iterable value for assignment.", node)

    def _assign_sequence_targets(
        self, targets: list[ast.expr], values: list[Any], node: ast.AST
    ) -> None:
        """Assign a concrete list of values to sequence targets (with optional *star)."""
        star_indices = [
            idx for idx, target in enumerate(targets) if isinstance(target, ast.Starred)
        ]

        if len(star_indices) > 1:
            raise EvalError(
                "Multiple starred expressions in assignment targets are not supported.",
                node,
            )

        if not star_indices:
            if len(targets) != len(values):
                raise EvalError(
                    f"Expected {len(targets)} values to unpack, but got {len(values)}.",
                    node,
                )
            for target, value in zip(targets, values):
                self._handle_destructuring_assignment(target, value)
            return

        star_index = star_indices[0]
        leading_targets = targets[:star_index]
        trailing_targets = targets[star_index + 1 :]
        required = len(leading_targets) + len(trailing_targets)

        if len(values) < required:
            raise EvalError(
                f"Not enough values to unpack (expected at least {required}, "
                f"got {len(values)}).",
                node,
            )

        leading_values = values[: len(leading_targets)]
        trailing_values = values[len(values) - len(trailing_targets) :]
        middle_values = values[
            len(leading_targets) : len(values) - len(trailing_targets)
        ]

        for target, value in zip(leading_targets, leading_values):
            self._handle_destructuring_assignment(target, value)

        self._assign_starred_target(targets[star_index], middle_values)

        for target, value in zip(trailing_targets, trailing_values):
            self._handle_destructuring_assignment(target, value)

    def _assign_starred_target(
        self, star_target: ast.Starred, values: list[Any]
    ) -> None:
        """Assign the collected middle values to a starred target."""
        # Star targets should always receive a list to mirror Python semantics.
        self._handle_destructuring_assignment(star_target.value, values)

    def _get_target_and_value(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise EvalError("Assignment must have exactly one target.", node)
        target = node.targets[0]
        value = node.value
        self._handle_destructuring_assignment(target, value)

    def visit(self, node: ast.AST) -> Any:
        """Override visit to add timeout checking on every AST node visit."""
        self._check_timeout()
        return super().visit(node)

    def add_sub_agent_time(self, duration: float) -> None:
        """Add time spent in sub-agent calls to be deducted from timeout."""
        self._sub_agent_time += duration

    def _check_timeout(self) -> None:
        """Check if execution has exceeded time limit."""
        # Calculate elapsed time, excluding sub-agent time
        current_time = time.time()
        elapsed = (current_time - self._start_time) - self._sub_agent_time

        if elapsed > self._timeout_seconds:
            raise EvalError(
                f"Program execution timed out after {self._timeout_seconds:.1f} seconds. "
                f"Consider optimizing your code or reducing computational complexity.",
                None,
            )

    def generic_visit(self, node: ast.AST) -> None:
        """
        Called for nodes that don't have a specific `visit_` method.
        This override prevents visiting children of unhandled nodes.
        """
        node_type = type(node).__name__

        # Provide specific helpful error messages for common unsupported features
        if isinstance(node, ast.Nonlocal):
            var_names = ", ".join(node.names)
            raise EvalError(
                f"The 'nonlocal' statement is not supported. "
                f"Consider using return values, object attributes, or mutable containers "
                f"instead of modifying '{var_names}' in the enclosing scope.",
                node,
            )
        elif isinstance(node, ast.Global):
            var_names = ", ".join(node.names)
            raise EvalError(
                f"The 'global' statement is not supported. "
                f"Variables '{var_names}' cannot be declared as global in the sandbox.",
                node,
            )
        elif isinstance(node, ast.Yield):
            raise EvalError(
                "Generator functions with 'yield' are not supported. "
                "Consider using regular functions that return lists or other data structures.",
                node,
            )
        elif isinstance(node, ast.YieldFrom):
            raise EvalError(
                "Generator functions with 'yield from' are not supported. "
                "Consider using regular functions that return lists or other data structures.",
                node,
            )
        elif isinstance(node, ast.Await):
            raise EvalError(
                "Async/await syntax is not supported. "
                "Consider using synchronous code patterns instead.",
                node,
            )
        elif isinstance(node, ast.AsyncFunctionDef):
            raise EvalError(
                "Async function definitions are not supported. "
                "Use regular 'def' function definitions instead.",
                node,
            )
        elif isinstance(node, ast.AsyncWith):
            raise EvalError(
                "Async context managers ('async with') are not supported. "
                "Use regular 'with' statements instead.",
                node,
            )
        elif isinstance(node, ast.AsyncFor):
            raise EvalError(
                "Async for loops ('async for') are not supported. "
                "Use regular 'for' loops instead.",
                node,
            )
        else:
            # Generic fallback for other unsupported nodes
            raise EvalError(f"AST node type '{node_type}' is not supported.", node)
