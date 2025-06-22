import ast
from typing import Any

from .base import BaseEvaluator
from .binop import OPERATOR_MAP
from .error import EvalError


class StatementEvaluator(BaseEvaluator):
    """A mixin for evaluating statement nodes."""

    def _resolve_subscript_for_mutation(
        self, node: ast.Subscript
    ) -> tuple[str, Any, Any, Any]:
        """
        Resolves a (potentially nested) subscript target for mutation.

        Returns a tuple of:
        - root_name: The name of the variable in the state.
        - root_container: The direct container to be mutated.
        - container_to_modify: The direct container to be mutated.
        - final_key: The key/index for the final assignment.
        """
        keys = []
        curr = node
        while isinstance(curr, ast.Subscript):
            keys.append(curr.slice)
            curr = curr.value

        if not isinstance(curr, ast.Name):
            raise EvalError(
                "Indexed assignment must start with a named variable.", node
            )

        keys.reverse()
        root_name = curr.id

        root_container = self.state.get(root_name)
        if root_container is None:
            raise EvalError(f"Name '{root_name}' is not defined.", node)

        container_to_modify = root_container
        for key_node in keys[:-1]:
            key = self.visit(key_node)
            try:
                container_to_modify = container_to_modify[key]
            except (KeyError, IndexError):
                raise EvalError(
                    f"Cannot resolve key {key} in nested structure.", key_node
                )
            except TypeError:
                raise EvalError("This object is not subscriptable.", key_node)

        final_key = self.visit(keys[-1])

        if not isinstance(container_to_modify, (dict, list)):
            raise EvalError(
                "Indexed assignment is currently only supported for dictionaries and lists.",
                node,
            )

        return (root_name, root_container, container_to_modify, final_key)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles assignment statements."""
        value = self.visit(node.value)

        for target in node.targets:
            if isinstance(target, (ast.Name, ast.Tuple)):
                if isinstance(target, ast.Tuple) and len(node.targets) > 1:
                    raise EvalError(
                        "Destructuring cannot be part of a chained assignment.", node
                    )
                self._handle_destructuring_assignment(target, value)
            elif isinstance(target, ast.Subscript):
                root_name, root_container, container, key = (
                    self._resolve_subscript_for_mutation(target)
                )
                try:
                    container[key] = value
                except IndexError:
                    raise EvalError("List assignment index out of range.", target)

                self.state.set(root_name, root_container)
            else:
                raise EvalError(
                    "This type of assignment target is not supported.", node
                )

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Handles augmented assignment statements like '+='."""
        op_func = OPERATOR_MAP.get(type(node.op))
        if not op_func:
            raise EvalError(f"Operator {type(node.op).__name__} not supported.", node)

        rhs_value = self.visit(node.value)

        # Step 1: Define how to get the current value and how to set the new one,
        # based on the target type (Name vs. Subscript).
        if isinstance(node.target, ast.Name):
            name = node.target.id
            current_value = self.state.get(name)
            if current_value is None and name not in self.state:
                raise EvalError(f"Name '{name}' is not defined.", node)

            def setter(value):
                self.state.set(name, value)

        elif isinstance(node.target, ast.Subscript):
            (
                root_name,
                root_container,
                container,
                key,
            ) = self._resolve_subscript_for_mutation(node.target)
            try:
                current_value = container[key]
            except (KeyError, IndexError):
                raise EvalError(
                    "Key or index not found for augmented assignment.", node.target
                )

            def setter(value):
                container[key] = value
                self.state.set(root_name, root_container)

        else:
            raise EvalError(
                "Augmented assignment to this target type is not supported.",
                node,
            )

        # Step 2: Perform the operation and call the setter. This part is now
        # independent of the target type.
        try:
            new_value = op_func(current_value, rhs_value)
            setter(new_value)
        except Exception as e:
            raise EvalError(f"Failed to execute operation: {e}", node, cause=e)
