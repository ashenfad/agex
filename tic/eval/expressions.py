import ast
from typing import Any

from ..eval.utils import get_allowed_attributes_for_instance
from .base import BaseEvaluator
from .builtins import BUILTINS
from .error import EvalError
from .objects import TicModule, TicObject


class ExpressionEvaluator(BaseEvaluator):
    """A mixin for evaluating expression nodes."""

    def _resolve_name(self, name: str, node: ast.AST) -> Any:
        """Resolve a name by checking builtins, state, and agent registries."""
        # 1. Check builtins
        if name in BUILTINS:
            return BUILTINS[name]

        # 2. Check the current execution state
        value = self.state.get(name)
        if value is not None or name in self.state:
            return value

        # 3. Check agent function registry
        if name in self.agent.fn_registry:
            from .functions import NativeFunction

            spec = self.agent.fn_registry[name]
            return NativeFunction(name=name, fn=spec.fn)

        # 4. Check agent class registry
        if name in self.agent.cls_registry:
            return self.agent.cls_registry[name].cls

        raise EvalError(f"Name '{name}' is not defined.", node)

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Handles literal values like numbers, strings, True, False, None."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        """Handles variable lookups."""
        return self._resolve_name(node.id, node)

    def visit_List(self, node: ast.List) -> list:
        """Handles list literals."""
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> tuple:
        """Handles tuple literals."""
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Set(self, node: ast.Set) -> set:
        """Handles set literals."""
        return {self.visit(elt) for elt in node.elts}

    def visit_Dict(self, node: ast.Dict) -> dict:
        """Handles dict literals."""
        return {
            self.visit(k): self.visit(v)
            for k, v in zip(node.keys, node.values)
            if k is not None
        }

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        """Handles boolean logic with short-circuiting ('and', 'or')."""
        if isinstance(node.op, ast.And):
            for value_node in node.values:
                result = self.visit(value_node)
                if not result:
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            for value_node in node.values:
                result = self.visit(value_node)
                if result:
                    return result
            return result
        else:
            raise EvalError(f"Unsupported boolean operator: {type(node.op)}", node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Handles ternary expressions like `a if condition else b`."""
        if self.visit(node.test):
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Handles attribute access like `obj.x`."""
        obj = self.visit(node.value)

        # Sandboxed TicObjects have their own logic
        if isinstance(obj, TicObject):
            return obj.getattr(node.attr)

        # Allow access to module attributes
        if isinstance(obj, TicModule):
            return getattr(obj, node.attr)

        # Check for registered host classes and whitelisted methods
        allowed_attrs = get_allowed_attributes_for_instance(self.agent, obj)
        if node.attr in allowed_attrs:
            return getattr(obj, node.attr)

        raise EvalError(
            f"'{type(obj).__name__}' object has no attribute '{node.attr}'", node
        )

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        """Handles subscript access like `d['key']` or `l[0]` or `l[1:5]`."""
        container = self.visit(node.value)

        if isinstance(node.slice, ast.Slice):
            lower = self.visit(node.slice.lower) if node.slice.lower else None
            upper = self.visit(node.slice.upper) if node.slice.upper else None
            step = self.visit(node.slice.step) if node.slice.step else None
            key = slice(lower, upper, step)
        else:
            key = self.visit(node.slice)

        try:
            return container[key]
        except (KeyError, IndexError):
            raise EvalError(f"Key or index not found: {key}", node)
        except TypeError:
            raise EvalError(
                "This object is not subscriptable or does not support slicing.", node
            )
