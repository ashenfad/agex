import ast
from typing import Any

from ..eval.utils import get_allowed_attributes_for_instance
from .base import BaseEvaluator
from .builtins import BUILTINS
from .error import EvalError
from .objects import TicInstance, TicModule, TicObject
from .user_errors import TicAttributeError


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
        """Handles attribute access like 'obj.attr'."""
        value = self.visit(node.value)

        # Handle TicModule with JIT resolution (implemented below)

        attr_name = node.attr

        # Sandboxed TicObjects have their own logic
        if isinstance(value, (TicObject, TicInstance)):
            return value.getattr(attr_name)

        # Handle TicModule attribute access with JIT resolution
        if isinstance(value, TicModule):
            # Look up the real module from the agent's registry
            reg_module = self.agent.importable_modules.get(value.name)
            if not reg_module:
                raise TicAttributeError(
                    f"Module '{value.name}' is not registered", node
                )

            # Get the attribute from the real module
            try:
                real_attr = getattr(reg_module.module, attr_name)
            except AttributeError:
                raise TicAttributeError(
                    f"module '{value.name}' has no attribute '{attr_name}'", node
                )

            # If the attribute is a module, return a new TicModule
            # Otherwise, return the actual value
            import types

            if isinstance(real_attr, types.ModuleType):
                return TicModule(name=f"{value.name}.{attr_name}")

            return real_attr

        # Check for registered host classes and whitelisted methods
        allowed_attrs = get_allowed_attributes_for_instance(self.agent, value)
        if attr_name in allowed_attrs:
            try:
                return getattr(value, attr_name)
            except AttributeError:
                raise TicAttributeError(
                    f"'{type(value).__name__}' object has no attribute '{attr_name}'",
                    node,
                )

        raise TicAttributeError(
            f"'{type(value).__name__}' object has no attribute '{attr_name}'", node
        )

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
        except (KeyError, IndexError):
            raise EvalError(f"Key or index not found: {key}", node)
        except TypeError:
            raise EvalError(
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
