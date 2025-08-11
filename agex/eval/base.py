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
        if isinstance(target_node, ast.Name):
            self.state.set(target_node.id, value)
        elif isinstance(target_node, ast.Tuple):
            if not hasattr(value, "__iter__"):
                raise EvalError(
                    "Cannot unpack non-iterable value for assignment.", target_node
                )

            targets = target_node.elts
            try:
                values = list(value)
            except TypeError:
                raise EvalError(
                    "Cannot unpack non-iterable value for assignment.", target_node
                )

            if len(targets) != len(values):
                raise EvalError(
                    f"Expected {len(targets)} values to unpack, but got {len(values)}.",
                    target_node,
                )

            for t, v in zip(targets, values):
                # Recurse to handle nested destructuring
                self._handle_destructuring_assignment(t, v)
        else:
            raise EvalError("Assignment target must be a name or a tuple.", target_node)

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
