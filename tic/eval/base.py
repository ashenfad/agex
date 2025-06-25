import ast
from typing import Any

from tic.agent import Agent
from tic.state.core import State

from .error import EvalError
from .objects import TicModule


class BaseEvaluator(ast.NodeVisitor):
    """A base class for evaluators, holding shared state."""

    def __init__(self, agent: "Agent", state: "State"):
        self.agent = agent
        self.state = state
        self.source_code: str | None = None

    def _create_tic_module(self, module_name: str) -> TicModule:
        """Creates and populates a sandboxed TicModule from the agent's registry."""
        reg_module = self.agent.importable_modules.get(module_name)

        if not reg_module:
            raise EvalError(
                f"Module '{module_name}' is not registered or whitelisted.", node=None
            )

        # Create a sandboxed module object
        tic_module = TicModule(name=module_name)
        for fn_name in reg_module.fns.keys():
            setattr(tic_module, fn_name, getattr(reg_module.module, fn_name))
        for const_name in reg_module.consts.keys():
            setattr(tic_module, const_name, getattr(reg_module.module, const_name))
        for cls_name, reg_class in reg_module.classes.items():
            setattr(tic_module, cls_name, reg_class.cls)

        return tic_module

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

    def generic_visit(self, node: ast.AST) -> None:
        """
        Called for nodes that don't have a specific `visit_` method.
        This override prevents visiting children of unhandled nodes.
        """
        raise NotImplementedError(f"AST node not supported: {type(node).__name__}")
