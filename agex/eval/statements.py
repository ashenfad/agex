import ast
from abc import ABC, abstractmethod
from typing import Any

from agex.agent.datatypes import _AgentExit
from agex.eval.builtins import dataclass
from agex.eval.functions import UserFunction, _ReturnException
from agex.eval.objects import BoundInstanceObject
from agex.eval.user_errors import (
    AgexAttributeError,
    AgexError,
    AgexIndexError,
    AgexKeyError,
    AgexNameError,
    AgexTypeError,
    AgexValueError,
)
from agex.state.scoped import Scoped

from .base import BaseEvaluator
from .binop import OPERATOR_MAP
from .error import EvalError
from .objects import AgexClass, AgexDataClass, AgexInstance, AgexObject
from .safe import check_assignment_safety


def _handle_assignment_exceptions(
    operation, node: ast.AST, operation_name: str = "Operation"
):
    """
    Helper function to handle common exceptions during assignment/deletion/access operations.

    Args:
        operation: A callable that performs the actual operation
        node: The AST node for error reporting
        operation_name: Human-readable name for the operation (e.g., "Assignment", "Deletion", "Access")
    """
    try:
        return operation()
    except AgexError:
        # Let user-facing errors bubble up unchanged (they already have line/col info)
        raise
    except IndexError:
        if "assignment" in operation_name.lower():
            raise AgexIndexError("List assignment index out of range.", node)
        elif "deletion" in operation_name.lower():
            raise AgexIndexError("List deletion index out of range.", node)
        else:
            raise AgexIndexError("List index out of range.", node)
    except KeyError as e:
        # Extract key from exception if available
        key = getattr(e, "args", [None])[0] if e.args else "unknown"
        raise AgexKeyError(f"Key '{key}' not found.", node)
    except AttributeError as e:
        raise AgexAttributeError(str(e), node)
    except ValueError as e:
        # Handle pandas and other ValueError cases
        raise AgexValueError(str(e), node) from e
    except TypeError as e:
        # Handle type errors during assignment
        raise AgexTypeError(str(e), node) from e
    except Exception as e:
        # Catch-all for other assignment errors, wrap with line/col info
        from .error import EvalError

        raise EvalError(f"{operation_name} error: {e}", node, cause=e)


class AssignmentTarget(ABC):
    """An abstract base class representing a resolved target for mutation."""

    @abstractmethod
    def get_value(self) -> Any:
        """Gets the current value of the target."""
        ...

    def set_value(self, value: Any):
        """Sets a new value for the target, checking pickle safety first."""
        safe_value = check_assignment_safety(value)
        self._do_set_value(safe_value)

    @abstractmethod
    def _do_set_value(self, value: Any):
        """Subclasses implement the actual assignment logic."""
        ...

    @abstractmethod
    def del_value(self):
        """Deletes the target."""
        ...


class NameTarget(AssignmentTarget):
    """Represents assignment to a variable name."""

    def __init__(self, evaluator: "BaseEvaluator", name: str):
        self._evaluator = evaluator
        self._name = name

    def get_value(self) -> Any:
        if self._name not in self._evaluator.state:
            raise AgexNameError(f"name '{self._name}' is not defined")
        return self._evaluator.state.get(self._name)

    def _do_set_value(self, value: Any):
        self._evaluator.state.set(self._name, value)

    def del_value(self):
        if self._name not in self._evaluator.state:
            raise AgexNameError(f"name '{self._name}' is not defined")
        self._evaluator.state.remove(self._name)


class TransientNameTarget(AssignmentTarget):
    """Represents assignment to a transient variable name (bypasses pickle safety)."""

    def __init__(self, evaluator: "BaseEvaluator", name: str):
        self._evaluator = evaluator
        self._name = name

    def get_value(self) -> Any:
        if self._name not in self._evaluator.state:
            raise AgexNameError(f"name '{self._name}' is not defined")
        return self._evaluator.state.get(self._name)

    def set_value(self, value: Any):
        """Set value directly without pickle safety check."""
        self._evaluator.state.set(self._name, value)

    def _do_set_value(self, value: Any):
        """This should not be called for transient targets."""
        self._evaluator.state.set(self._name, value)

    def del_value(self):
        if self._name not in self._evaluator.state:
            raise AgexNameError(f"name '{self._name}' is not defined")
        self._evaluator.state.remove(self._name)


class AttributeTarget(AssignmentTarget):
    """Represents assignment to an object attribute."""

    def __init__(self, obj: Any, attr_name: str, node: ast.AST):
        # Allow attribute assignment on any object that supports it
        # Check if the object supports attribute assignment
        supports_assignment = (
            hasattr(obj, attr_name)  # Attribute already exists
            or hasattr(obj, "__dict__")  # Object has a __dict__ for new attributes
            or hasattr(obj, "__setattr__")  # Object defines custom __setattr__
        )

        if not supports_assignment:
            raise AgexTypeError(
                f"'{type(obj).__name__}' object does not support attribute assignment.",
                node,
            )

        self._obj = obj
        self._attr_name = attr_name
        self._node = node

    def get_value(self) -> Any:
        def do_access():
            # Handle AgexObject, AgexInstance, and BoundInstanceObject with their special methods
            if isinstance(self._obj, (AgexObject, AgexInstance, BoundInstanceObject)):
                return self._obj.getattr(self._attr_name)
            else:
                # Handle regular Python objects
                return getattr(self._obj, self._attr_name)

        try:
            return _handle_assignment_exceptions(
                do_access, self._node, "Attribute access"
            )
        except AgexAttributeError:
            # Provide a more specific error message for attribute access
            raise AgexAttributeError(
                f"'{type(self._obj).__name__}' object has no attribute '{self._attr_name}'",
                self._node,
            )

    def _do_set_value(self, value: Any):
        def do_assignment():
            # Handle AgexObject, AgexInstance, and BoundInstanceObject with their special methods
            if isinstance(self._obj, (AgexObject, AgexInstance, BoundInstanceObject)):
                self._obj.setattr(self._attr_name, value)
            else:
                # Handle regular Python objects
                setattr(self._obj, self._attr_name, value)

        _handle_assignment_exceptions(do_assignment, self._node, "Attribute assignment")

    def del_value(self):
        def do_deletion():
            # Handle AgexObject, AgexInstance, and BoundInstanceObject with their special methods
            if isinstance(self._obj, (AgexObject, AgexInstance, BoundInstanceObject)):
                self._obj.delattr(self._attr_name)
            else:
                # Handle regular Python objects
                delattr(self._obj, self._attr_name)

        _handle_assignment_exceptions(do_deletion, self._node, "Attribute deletion")


class SubscriptTarget(AssignmentTarget):
    """Represents assignment to a list index or dict key."""

    def __init__(self, evaluator, node: ast.Subscript):
        # This logic is complex and is largely migrated from the old
        # `_resolve_subscript_for_mutation` helper.
        self._evaluator = evaluator
        self._node = node

        keys = []
        curr: ast.AST = node
        while isinstance(curr, ast.Subscript):
            keys.append(evaluator.visit(curr.slice))
            curr = curr.value

        keys.reverse()
        self._container = evaluator.visit(curr)

        # To update state correctly, we need to find the root variable.
        self._root_name: str | None = None
        self._root_container = None
        temp_curr = curr
        if isinstance(temp_curr, ast.Attribute):
            while isinstance(temp_curr, ast.Attribute):
                temp_curr = temp_curr.value
            if isinstance(temp_curr, ast.Name):
                self._root_name = temp_curr.id
                self._root_container = evaluator.state.get(self._root_name)
        elif isinstance(temp_curr, ast.Name):
            self._root_name = temp_curr.id
            self._root_container = evaluator.state.get(self._root_name)

        # Traverse the keys to get to the final container
        self._final_key = keys.pop()
        for key in keys:
            try:
                self._container = self._container[key]
            except (KeyError, IndexError):
                raise AgexKeyError(
                    f"Cannot resolve key {key} in nested structure.", node
                )
            except TypeError:
                raise AgexTypeError("This object is not subscriptable.", node)

        if not hasattr(self._container, "__setitem__"):
            raise AgexTypeError(
                f"Object of type '{type(self._container).__name__}' does not support item assignment.",
                node,
            )

    def get_value(self) -> Any:
        def do_access():
            return self._container[self._final_key]

        return _handle_assignment_exceptions(do_access, self._node, "Subscript access")

    def _do_set_value(self, value: Any):
        def do_assignment():
            self._container[self._final_key] = value
            if self._root_name:
                self._evaluator.state.set(self._root_name, self._root_container)

        _handle_assignment_exceptions(do_assignment, self._node, "Subscript assignment")

    def del_value(self):
        def do_deletion():
            del self._container[self._final_key]
            if self._root_name:
                self._evaluator.state.set(self._root_name, self._root_container)

        _handle_assignment_exceptions(do_deletion, self._node, "Subscript deletion")


class StatementEvaluator(BaseEvaluator):
    """A mixin for evaluating statement nodes."""

    def _resolve_target(self, node: ast.expr) -> AssignmentTarget:
        """Resolves an expression node into a concrete AssignmentTarget."""
        if isinstance(node, ast.Name):
            # Check if we're in a transient scope and this is a transient variable
            from agex.state import is_live_root
            from agex.state.transient import TransientScope

            if (
                isinstance(self.state, TransientScope)
                and node.id in self.state._transient_vars
            ):
                return TransientNameTarget(self, node.id)

            # Use TransientNameTarget if the root state is live (skip pickle enforcement)
            if is_live_root(self.state):
                return TransientNameTarget(self, node.id)

            return NameTarget(self, node.id)
        if isinstance(node, ast.Attribute):
            obj = self.visit(node.value)
            return AttributeTarget(obj, node.attr, node)
        if isinstance(node, ast.Subscript):
            return SubscriptTarget(self, node)
        raise EvalError("This type of assignment target is not supported.", node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles assignment statements."""
        value = self.visit(node.value)

        for target_node in node.targets:
            if isinstance(target_node, ast.Tuple):
                # Destructuring assignment, e.g., `a, b = 1, 2`
                if len(node.targets) > 1:
                    raise EvalError(
                        "Destructuring cannot be part of a chained assignment.", node
                    )
                self._handle_destructuring_assignment(target_node, value)
            else:
                target = self._resolve_target(target_node)
                target.set_value(value)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Handles the 'del' statement."""
        for target_node in node.targets:
            target = self._resolve_target(target_node)
            target.del_value()

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
        target = self._resolve_target(node.target)

        try:
            current_value = target.get_value()
            new_value = op_func(current_value, rhs_value)
            target.set_value(new_value)
        except AgexError:
            # Let user-facing errors from the getter/setter propagate.
            raise
        except Exception as e:
            # Wrap any other errors (e.g., from the op_func) in an EvalError.
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

    def visit_With(self, node: ast.With) -> None:
        """Handles with statements for context managers and transient variables."""
        if len(node.items) != 1:
            raise EvalError(
                "Multiple context managers in a single 'with' statement are not supported. "
                "Use nested 'with' statements instead.",
                node,
            )

        with_item = node.items[0]
        context_obj = self.visit(with_item.context_expr)

        # Check if it's a traditional context manager
        is_context_manager = hasattr(context_obj, "__enter__") and hasattr(
            context_obj, "__exit__"
        )

        if is_context_manager:
            # Traditional context manager path
            self._handle_context_manager(node, with_item, context_obj)
        else:
            # Transient variable path - use the object directly
            self._handle_transient_with(node, with_item, context_obj)

    def _handle_context_manager(
        self, node: ast.With, with_item: ast.withitem, context_manager: Any
    ) -> None:
        """Handle traditional context managers with __enter__/__exit__."""
        # Call __enter__ and optionally bind the result to a variable
        try:
            enter_result = context_manager.__enter__()
        except Exception as enter_exc:
            # If __enter__ fails, we don't call __exit__
            raise EvalError(
                f"Error entering context manager: {enter_exc}",
                node,
                cause=enter_exc,
            )

        # If there's an 'as' clause, bind the result
        if with_item.optional_vars:
            if isinstance(with_item.optional_vars, ast.Name):
                self.state.set(with_item.optional_vars.id, enter_result)
            else:
                # Handle tuple unpacking in 'as' clause
                target = self._resolve_target(with_item.optional_vars)
                target.set_value(enter_result)

        # Execute the body
        exception_info = (None, None, None)

        try:
            for stmt in node.body:
                self.visit(stmt)
        except Exception as e:
            exception_info = (type(e), e, None)  # Simplified traceback

            # Let control flow exceptions pass through immediately
            if isinstance(e, (_ReturnException, _AgentExit)):
                # Still need to call __exit__ but don't suppress the exception
                try:
                    context_manager.__exit__(*exception_info)
                except Exception:
                    pass  # Ignore exceptions from __exit__ for control flow
                raise e

            # For other exceptions, let __exit__ decide whether to suppress
            try:
                suppress = context_manager.__exit__(*exception_info)
                if not suppress:
                    raise e
            except Exception as exit_exc:
                # If __exit__ raises an exception, that replaces the original
                raise exit_exc
        else:
            # No exception occurred, call __exit__ with None values
            try:
                context_manager.__exit__(None, None, None)
            except Exception as exit_exc:
                # If __exit__ raises an exception when no exception occurred, propagate it
                raise exit_exc

    def _handle_transient_with(
        self, node: ast.With, with_item: ast.withitem, context_obj: Any
    ) -> None:
        """Handle transient variables - objects used directly without context manager protocol."""
        from agex.state.transient import TransientScope

        # Determine which variables should be transient
        transient_vars = set()
        if with_item.optional_vars:
            if isinstance(with_item.optional_vars, ast.Name):
                transient_vars.add(with_item.optional_vars.id)
            else:
                # For complex assignments, we'll need to be more conservative
                # For now, just handle simple names
                pass

        # Create a transient scope for the with block
        original_state = self.state
        self.state = TransientScope(original_state, transient_vars)

        try:
            # If there's an 'as' clause, bind the object directly to the variable
            if with_item.optional_vars:
                if isinstance(with_item.optional_vars, ast.Name):
                    # This assignment bypasses pickle safety for transient variables
                    self.state.set(with_item.optional_vars.id, context_obj)
                else:
                    # Handle tuple unpacking in 'as' clause
                    target = self._resolve_target(with_item.optional_vars)
                    target.set_value(context_obj)

            # Execute the body with transient scope
            for stmt in node.body:
                self.visit(stmt)

        finally:
            # Restore original state - transient variables are automatically cleaned up
            self.state = original_state

    def visit_Import(self, node: ast.Import) -> None:
        """Handles `import <module>` and `import <module> as <alias>`."""
        for alias in node.names:
            module_name_to_find = alias.name
            try:
                tic_module = self.resolver.resolve_module(module_name_to_find, node)
            except EvalError as e:
                e.node = node  # Add location info to the error
                raise

            # The name used in the agent's code, e.g., `m` in `import math as m`
            import_name = alias.asname or module_name_to_find
            self.state.set(import_name, tic_module)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handles `from <module> import <name>`."""
        module_name_to_find = node.module
        if not module_name_to_find:
            raise EvalError("Relative imports are not supported.", node)

        # Special case: allow `from dataclasses import dataclass` as a no-op
        # because we provide our own built-in `dataclass` object.
        if module_name_to_find == "dataclasses":
            is_just_dataclass_import = all(
                alias.name == "dataclass" and alias.asname is None
                for alias in node.names
            )
            if is_just_dataclass_import:
                return  # Silently ignore and succeed.

        for alias in node.names:
            name_to_import = alias.name
            target_name = alias.asname or name_to_import

            member = self.resolver.import_from(
                module_name_to_find, name_to_import, node
            )
            self.state.set(target_name, member)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """
        Handles annotated assignments (e.g., `x: int`).

        In the context of a class body, this is used to collect field names
        for dataclasses. We don't actually evaluate the annotation, we just
        record the variable name. In other contexts, it's a no-op.
        """
        # This visitor is primarily for dataclass parsing. The logic to use
        # the result is within visit_ClassDef. Outside of a class, it does nothing.
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handles class definition statements, supporting both regular and dataclasses."""
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

        if node.bases or node.keywords:
            raise EvalError(
                "Inheritance and other advanced class features are not supported.", node
            )

        # 2. Dispatch to the correct handler based on decorator.
        if is_dataclass:
            self._create_dataclass(node)
        else:
            self._create_regular_class(node)

    def _create_dataclass(self, node: ast.ClassDef):
        """Creates a AgexDataClass from a class definition node."""
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

        cls_obj = AgexDataClass(name=node.name, fields={f: None for f in fields})
        self.state.set(node.name, cls_obj)

    def _create_regular_class(self, node: ast.ClassDef):
        """Creates a AgexClass for a regular class definition."""
        from agex.eval.core import Evaluator

        # To execute the class body in isolation, we create a new evaluator
        # with its own temporary, scoped state.
        class_exec_state = Scoped(self.state)
        class_evaluator = Evaluator(
            agent=self.agent,
            state=class_exec_state,
            source_code=self.source_code,
            # Class definitions inherit the agent's timeout
        )

        # Execute the body of the class using the new evaluator.
        for stmt in node.body:
            class_evaluator.visit(stmt)

        # Extract methods (UserFunctions) from the temporary state's local scope.
        methods = {
            name: value
            for name, value in class_exec_state._local_store.items()
            if isinstance(value, UserFunction)
        }

        # Create the AgexClass object.
        cls = AgexClass(name=node.name, methods=methods)

        # Assign the new class to its name in the *main* state.
        self.state.set(node.name, cls)
