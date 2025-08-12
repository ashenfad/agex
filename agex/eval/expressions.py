import ast
from typing import Any

from .base import BaseEvaluator
from .error import EvalError
from .loops import _safe_bool_eval
from .user_errors import AgexIndexError, AgexKeyError, AgexTypeError


class ExpressionEvaluator(BaseEvaluator):
    """A mixin for evaluating expression nodes."""

    def visit_Constant(self, node: ast.Constant) -> Any:
        """Handles literal values like numbers, strings, True, False, None."""
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        """Handles variable lookups."""
        return self.resolver.resolve_name(node.id, self.state, node)

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
                if not _safe_bool_eval(result, value_node, "Boolean 'and' operation"):
                    return result
            return result
        elif isinstance(node.op, ast.Or):
            for value_node in node.values:
                result = self.visit(value_node)
                if _safe_bool_eval(result, value_node, "Boolean 'or' operation"):
                    return result
            return result
        else:
            raise EvalError(f"Unsupported boolean operator: {type(node.op)}", node)

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Handles ternary expressions like `a if condition else b`."""
        test_result = self.visit(node.test)
        if _safe_bool_eval(test_result, node.test, "Ternary expression condition"):
            return self.visit(node.body)
        else:
            return self.visit(node.orelse)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        """Handles attribute access like 'obj.attr'."""
        value = self.visit(node.value)
        return self.resolver.resolve_attribute(value, node.attr, node)

    def visit_Slice(self, node: ast.Slice) -> slice:
        """Handles slice objects."""
        lower = self.visit(node.lower) if node.lower else None
        upper = self.visit(node.upper) if node.upper else None
        step = self.visit(node.step) if node.step else None
        return slice(lower, upper, step)

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
        except KeyError:
            raise AgexKeyError(f"Key not found: {key}", node)
        except IndexError:
            raise AgexIndexError(f"Index out of range: {key}", node)
        except TypeError:
            raise AgexTypeError(
                "This object is not subscriptable or does not support slicing.", node
            )

    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        """Handles formatted values in f-strings."""
        value = self.visit(node.value)

        # First, apply conversion if any is specified (!s, !r, !a).
        if node.conversion == 115:  # !s
            value_to_format = str(value)
        elif node.conversion == 114:  # !r
            value_to_format = repr(value)
        elif node.conversion == 97:  # !a
            value_to_format = ascii(value)
        else:
            value_to_format = value

        if node.format_spec:
            # The format_spec is an expression (often a JoinedStr) that needs evaluation.
            format_spec = self.visit(node.format_spec)
            return format(value_to_format, format_spec)

        return str(value_to_format)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        """Handles f-strings by joining all the parts."""
        return "".join([self.visit(v) for v in node.values])
