import ast
from typing import Any

from .base import BaseEvaluator
from .call import BUILTINS
from .error import EvalError
from .objects import TicModule, TicObject


class ExpressionEvaluator(BaseEvaluator):
    """A mixin for evaluating expression nodes."""

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Handles literal values like numbers, strings, True, False, None."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        """Handles variable lookups."""
        if node.id in BUILTINS:
            return BUILTINS[node.id]

        value = self.state.get(node.id)
        if value is None and node.id not in self.state:
            raise EvalError(f"Name '{node.id}' is not defined.", node)
        return value

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
        if isinstance(obj, TicObject):
            return obj.getattr(node.attr)
        if isinstance(obj, TicModule):
            return getattr(obj, node.attr)

        raise EvalError(
            f"Attribute access is not supported for type '{type(obj).__name__}'.", node
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
