import ast
import time
from typing import Any

from agex.agent.base import BaseAgent
from agex.state.core import State

from .error import EvalError
from .objects import AgexModule


class BaseEvaluator(ast.NodeVisitor):
    """A base class for evaluators, holding shared state."""

    def __init__(
        self, agent: "BaseAgent", state: "State", timeout_seconds: float = 5.0
    ):
        self.agent = agent
        self.state = state
        self.source_code: str | None = None
        self._start_time = time.time()
        self._timeout_seconds = timeout_seconds
        self._operation_count = 0

    def _create_agex_module(self, module_name: str) -> AgexModule:
        """Creates a sandboxed AgexModule from the agent's registry."""
        reg_module = self.agent.importable_modules.get(module_name)

        if not reg_module:
            raise EvalError(
                f"Module '{module_name}' is not registered or whitelisted.", node=None
            )

        # Create a simple AgexModule with the name
        # JIT resolution happens in expressions.py when attributes are accessed
        return AgexModule(name=module_name)

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

    def _check_timeout(self) -> None:
        """Check if execution has exceeded time or operation limits."""
        self._operation_count += 1

        # Check time-based timeout
        elapsed = time.time() - self._start_time
        if elapsed > self._timeout_seconds:
            raise EvalError(
                f"Program execution timed out after {self._timeout_seconds:.1f} seconds. "
                f"Consider optimizing your code or reducing computational complexity.",
                None,
            )

        # Check operation-based timeout (backup safety net)
        # This catches cases where time.time() calls are expensive or unreliable
        max_operations = max(
            1000000, int(self._timeout_seconds * 100000)
        )  # ~100000 ops/sec baseline
        if self._operation_count > max_operations:
            raise EvalError(
                f"Program exceeded maximum operation limit ({max_operations:,} operations). "
                f"This likely indicates an infinite loop or very inefficient algorithm.",
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
