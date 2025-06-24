import ast
from typing import Any

from tic.agent import _AgentExit
from tic.eval.builtins import dataclass
from tic.eval.functions import _ReturnException

from .base import BaseEvaluator
from .binop import OPERATOR_MAP
from .error import EvalError
from .objects import TicDataClass, TicModule, TicObject


class StatementEvaluator(BaseEvaluator):
    """A mixin for evaluating statement nodes."""

    def _resolve_subscript_for_mutation(
        self, node: ast.Subscript
    ) -> tuple[str | None, Any, Any, Any]:
        """
        Resolves a (potentially nested) subscript target for mutation.

        Returns a tuple of:
        - root_name: The name of the variable in the state, if any.
        - root_container: The top-level container from the state.
        - container_to_modify: The direct container to be mutated.
        - final_key: The key/index for the final assignment.
        """
        keys = []
        curr: ast.AST = node
        while isinstance(curr, ast.Subscript):
            keys.append(curr.slice)
            curr = curr.value

        keys.reverse()
        container_to_modify = self.visit(curr)

        # To update state correctly, we need to find the root variable.
        root_name: str | None = None
        root_container = None
        temp_curr = curr
        if isinstance(temp_curr, ast.Attribute):
            while isinstance(temp_curr, ast.Attribute):
                temp_curr = temp_curr.value
            if isinstance(temp_curr, ast.Name):
                root_name = temp_curr.id
                root_container = self.state.get(root_name)
        elif isinstance(temp_curr, ast.Name):
            root_name = temp_curr.id
            root_container = self.state.get(root_name)

        # Traverse the keys to get to the final container
        final_key_node = keys.pop()
        for key_node in keys:
            key = self.visit(key_node)
            try:
                container_to_modify = container_to_modify[key]
            except (KeyError, IndexError):
                raise EvalError(
                    f"Cannot resolve key {key} in nested structure.", key_node
                )
            except TypeError:
                raise EvalError("This object is not subscriptable.", key_node)

        final_key = self.visit(final_key_node)

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

                if root_name:
                    self.state.set(root_name, root_container)
            elif isinstance(target, ast.Attribute):
                obj = self.visit(target.value)
                if not isinstance(obj, TicObject):
                    raise EvalError(
                        "Attribute assignment is only supported for dataclass instances.",
                        target,
                    )
                obj.setattr(target.attr, value)
            else:
                raise EvalError(
                    "This type of assignment target is not supported.", node
                )

    def visit_Pass(self, node: ast.Pass) -> None:
        """Handles the 'pass' statement."""
        # The 'pass' statement does nothing.
        pass

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
                if root_name:
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

    def visit_Try(self, node: ast.Try) -> None:
        """Handles try...except...else...finally blocks."""
        # The 'finally' block must execute regardless of what happens.
        try:
            # We track if an exception was caught to decide if we run 'else'.
            exception_was_caught = False
            try:
                # Execute the main 'try' block.
                for stmt in node.body:
                    self.visit(stmt)
            except Exception as e:
                # An exception occurred.
                exception_was_caught = True

                # IMPORTANT: If this is an internal control-flow exception,
                # we must not let the user's code catch it. Re-raise it.
                if isinstance(e, (_ReturnException, _AgentExit)):
                    raise e

                # Find a matching 'except' handler in the user's code.
                matched_handler = None
                for handler in node.handlers:
                    # handler.type can be None for a bare 'except:'.
                    if handler.type is None:
                        matched_handler = handler
                        break

                    # Evaluate the exception type specified by the user.
                    exc_type_to_catch = self.visit(handler.type)

                    # Check if it's a valid type and if our error is an instance.
                    if isinstance(exc_type_to_catch, type) and isinstance(
                        e, exc_type_to_catch
                    ):
                        matched_handler = handler
                        break

                if matched_handler:
                    # If we found a handler, execute its body.
                    # If 'as' is used, set the exception instance in the state.
                    if matched_handler.name:
                        self.state.set(matched_handler.name, e)

                    for handler_stmt in matched_handler.body:
                        self.visit(handler_stmt)

                    # Clean up the 'as' variable if it was set.
                    if matched_handler.name:
                        self.state.remove(matched_handler.name)
                else:
                    # No matching handler was found, so re-raise the exception.
                    raise e

            # The 'else' block runs only if the 'try' block completed
            # without raising any exceptions.
            if not exception_was_caught:
                for else_stmt in node.orelse:
                    self.visit(else_stmt)

        finally:
            # The 'finally' block always runs.
            for final_stmt in node.finalbody:
                self.visit(final_stmt)

    def visit_Raise(self, node: ast.Raise) -> None:
        """Handles the 'raise' statement."""
        if node.exc:
            exc = self.visit(node.exc)
            if isinstance(exc, type) and issubclass(exc, BaseException):
                raise exc()
            if isinstance(exc, BaseException):
                raise exc
            raise EvalError(
                f"Can only raise exception classes or instances, not {type(exc).__name__}",
                node,
            )
        raise

    def visit_Import(self, node: ast.Import) -> None:
        """Handles `import <module>` and `import <module> as <alias>`."""
        for alias in node.names:
            module_name_to_find = alias.name
            reg_module = self.agent.importable_modules.get(module_name_to_find)

            if not reg_module:
                raise EvalError(
                    f"Module '{module_name_to_find}' is not registered or whitelisted.",
                    node,
                )

            # Create a sandboxed module object
            tic_module = TicModule(name=module_name_to_find)
            for fn_name in reg_module.fns.keys():
                setattr(tic_module, fn_name, getattr(reg_module.module, fn_name))
            for const_name in reg_module.consts.keys():
                setattr(tic_module, const_name, getattr(reg_module.module, const_name))
            for cls_name, reg_class in reg_module.classes.items():
                setattr(tic_module, cls_name, reg_class.cls)

            # The name used in the agent's code, e.g., `m` in `import math as m`
            import_name = alias.asname or module_name_to_find
            self.state.set(import_name, tic_module)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handles `from <module> import <name>`."""
        module_name_to_find = node.module
        if not module_name_to_find:
            raise EvalError("Relative imports are not supported.", node)

        reg_module = self.agent.importable_modules.get(module_name_to_find)
        if not reg_module:
            raise EvalError(
                f"Module '{module_name_to_find}' is not registered or whitelisted.",
                node,
            )

        for alias in node.names:
            name_to_import = alias.name
            import_as = alias.asname or name_to_import

            # Check fns, consts, and classes
            if name_to_import in reg_module.fns:
                obj = getattr(reg_module.module, name_to_import)
            elif name_to_import in reg_module.consts:
                obj = getattr(reg_module.module, name_to_import)
            elif name_to_import in reg_module.classes:
                obj = reg_module.classes[name_to_import].cls
            elif name_to_import == "*":
                raise EvalError("Wildcard imports are not supported.", node)
            else:
                raise EvalError(
                    f"Cannot import name '{name_to_import}' from module '{module_name_to_find}'.",
                    node,
                )
            self.state.set(import_as, obj)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handles class definitions (only dataclasses are supported)."""
        # 1. Check for the @dataclass decorator.
        is_dataclass = False
        if node.decorator_list:
            # For simplicity, we only support a single decorator, @dataclass.
            if len(node.decorator_list) > 1:
                raise EvalError(
                    "Only a single @dataclass decorator is supported.", node
                )
            decorator = self.visit(node.decorator_list[0])
            if decorator is dataclass:
                is_dataclass = True

        if not is_dataclass:
            raise EvalError(
                "Class definitions must use the @dataclass decorator.", node
            )

        # 2. Forbid inheritance.
        if node.bases or node.keywords:
            raise EvalError("Dataclass inheritance is not supported.", node)

        # 3. Extract fields and forbid methods.
        fields = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                if not isinstance(stmt.target, ast.Name):
                    raise EvalError("Dataclass fields must be simple names.", stmt)
                fields.append(stmt.target.id)
            elif isinstance(stmt, ast.FunctionDef):
                raise EvalError("Methods are not supported in dataclasses.", stmt)
            else:
                raise EvalError(
                    "Only annotated assignments (e.g., 'x: int') are allowed in dataclass bodies.",
                    stmt,
                )

        if not fields:
            raise EvalError("Dataclasses must define at least one field.", node)

        # 4. Create and store the TicDataClass object.
        cls_obj = TicDataClass(name=node.name, fields=fields)  # type: ignore
        self.state.set(node.name, cls_obj)
