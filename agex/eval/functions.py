import ast
import inspect
import time
import warnings
from dataclasses import dataclass, make_dataclass
from typing import Any, Callable

from kvit import Store

from agex.agent.base import get_agent_by_name, resolve_agent

from ..state.closure import LiveClosureState
from ..state.scoped import Scoped
from .analysis import get_free_variables
from .base import BaseEvaluator


class _ReturnException(Exception):
    """Internal exception to signal a return statement, carrying the return value."""

    def __init__(self, value: Any, node: ast.Return):
        self.value = value
        self.node = node


@dataclass
class NativeFunction:
    """Represents a native Python function available in the Tic environment."""

    name: str
    fn: Callable[..., Any]
    host_fs_access: bool = False
    network_access: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Directly call the wrapped native function.
        return self.fn(*args, **kwargs)

    # New unified execution hook used by the evaluator
    def execute(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        return self.fn(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Preserve important attributes from the wrapped function
        # This is especially important for dual-decorated functions
        # that have __agex_task_namespace__ attributes
        if hasattr(self.fn, name):
            return getattr(self.fn, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __deepcopy__(self, memo):
        # no deepcopy for native functions
        return self

    @property
    def __doc__(self):
        return self.fn.__doc__


@dataclass
class UserFunction:
    """Represents a user-defined function and its closure."""

    name: str
    args: ast.arguments
    body: list[ast.stmt]
    closure_state: Store  # A *reference* to the state where the function was defined.
    source_text: str | None = None
    agent_fingerprint: str | None = (
        None  # Fingerprint of the agent this function was defined in
    )
    agent_name: str | None = None  # Name of the agent (for fallback resolution)
    session: str = "default"  # Session where the function was defined
    # Pre-evaluated default values (captured at definition time, like Python)
    evaluated_defaults: list[Any] | None = None
    evaluated_kw_defaults: list[Any] | None = None

    # Ensure hashability for use in libraries that cache by callable (e.g., pandas.apply)
    def __hash__(self) -> int:  # type: ignore[override]
        # Identity-based hash keeps semantics simple and avoids mutable field issues
        return hash(id(self))

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return self is other

    @property
    def __signature__(self) -> inspect.Signature:
        """Convert AST arguments to inspect.Signature for compatibility with inspect.signature()"""
        parameters = []

        # Convert positional arguments
        for arg in self.args.args:
            parameters.append(
                inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            )

        # Convert keyword-only arguments
        for arg in self.args.kwonlyargs:
            parameters.append(
                inspect.Parameter(arg.arg, inspect.Parameter.KEYWORD_ONLY)
            )

        # Convert *args if present
        if self.args.vararg:
            parameters.append(
                inspect.Parameter(
                    self.args.vararg.arg, inspect.Parameter.VAR_POSITIONAL
                )
            )

        # Convert **kwargs if present
        if self.args.kwarg:
            parameters.append(
                inspect.Parameter(self.args.kwarg.arg, inspect.Parameter.VAR_KEYWORD)
            )

        return inspect.Signature(parameters)

    def __post_init__(self):
        """Set standard function attributes."""
        self.__name__ = self.name
        self.__qualname__ = self.name

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.agent_fingerprint:
            raise RuntimeError(
                "UserFunction cannot be called directly without an Agent context."
            )
        # No source code available from fingerprint
        return self.execute(list(args), kwargs, None, parent_evaluator=None)

    def execute(
        self, args: list, kwargs: dict, source_code: str | None, parent_evaluator=None
    ):
        """Execute the function with a new evaluator."""
        import asyncio

        from agex.eval.arguments import bind_arguments
        from agex.eval.core import Evaluator

        exec_state = Scoped(self.closure_state)

        if not self.agent_fingerprint:
            raise RuntimeError("Cannot execute function without an agent context.")

        # Resolve agent from fingerprint, with fallback by name
        agent = None
        try:
            agent = resolve_agent(self.agent_fingerprint)
        except RuntimeError:
            # Fingerprint not found - try fallback by agent name
            if self.agent_name:
                agent = get_agent_by_name(self.agent_name)
            if agent:
                warnings.warn(
                    f"User function '{self.name}' was created by agent with fingerprint "
                    f"'{self.agent_fingerprint[:8]}...' which is no longer registered. "
                    f"Falling back to agent '{agent.name}' by name. "
                    f"This may happen after config changes (primer, registrations).",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    f"Cannot execute user function '{self.name}': "
                    f"No agent found with fingerprint '{self.agent_fingerprint[:8]}...' "
                    f"or name '{self.agent_name}'."
                )

        # Create evaluator with timeout context from parent if available
        if parent_evaluator is not None:
            # Inherit timeout context from parent evaluator
            evaluator = Evaluator(
                agent=agent,
                state=exec_state,
                source_code=source_code,
                eval_timeout_seconds=parent_evaluator._eval_timeout_seconds,
                start_time=parent_evaluator._start_time,
                sub_agent_time=parent_evaluator._sub_agent_time,
                session=parent_evaluator.session,
            )
        else:
            # Fresh timeout budget (for direct calls)
            # Try to get the running event loop for async bridging (e.g., NiceGUI callbacks)
            try:
                main_loop = asyncio.get_running_loop()
            except RuntimeError:
                main_loop = None

            evaluator = Evaluator(
                agent=agent,
                state=exec_state,
                source_code=source_code,
                eval_timeout_seconds=agent.eval_timeout_seconds,
                session=self.session,  # Use session from when function was defined
                main_loop=main_loop,
            )

        # Check if this is a method call with context for super()
        if args and hasattr(args[0], "_method_context"):
            method_context = getattr(args[0], "_method_context", None)
            if method_context:
                defining_class, instance = method_context
                evaluator.current_method_class = defining_class
                evaluator.current_self = instance

        bound_args = bind_arguments(
            self.name,
            self.args,
            args,
            kwargs,
            eval_fn=evaluator.visit,
            evaluated_defaults=self.evaluated_defaults,
            evaluated_kw_defaults=self.evaluated_kw_defaults,
        )
        for name, value in bound_args.items():
            exec_state.set(name, value)

        # Set up VFS context if agent has filesystem configured
        # This ensures file operations in callbacks (e.g., NiceGUI button clicks)
        # route to VFS instead of real filesystem
        fs_context = None
        if agent._fs_config:
            from agex.fs import with_fs_context

            fs_backend, _ = agent._get_fs_backend(evaluator.session)
            fs_context = with_fs_context(fs_backend, defer_snapshots=False)

        try:
            if fs_context is not None:
                fs_context.__enter__()

            for node in self.body:
                evaluator.visit(node)
            return None
        except _ReturnException as e:
            return e.value
        finally:
            if fs_context is not None:
                fs_context.__exit__(None, None, None)


def create_inputs_dataclass_from_ast_args(
    task_name: str, args: ast.arguments, use_generic_types: bool = False
) -> type:
    """
    Create a dataclass for task inputs from AST arguments.

    Args:
        task_name: Name of the task function
        args: AST arguments from function definition
        use_generic_types: If True, use Any for all types (for UserFunction conversion)

    Returns:
        Dynamically created dataclass type
    """
    if not args.args:
        # No inputs - return empty dataclass
        to_camel_case = lambda snake_str: "".join(
            x.capitalize() for x in snake_str.lower().split("_")
        )
        dataclass_name = f"{to_camel_case(task_name)}Inputs"
        return make_dataclass(dataclass_name, [])

    # Build field specifications
    fields = []
    for arg in args.args:
        param_name = arg.arg
        # Use Any for generic types (UserFunction case) or infer from annotation
        param_type = Any if use_generic_types else object  # Can be enhanced later
        fields.append((param_name, param_type))

    # Handle defaults if present
    if args.defaults:
        num_defaults = len(args.defaults)
        num_params = len(args.args)
        defaults_start = num_params - num_defaults

        # Update fields with defaults
        for i, default_value in enumerate(args.defaults):
            field_index = defaults_start + i
            param_name = args.args[field_index].arg
            # Replace the field to include default
            fields[field_index] = (param_name, fields[field_index][1], default_value)

    # Create the dataclass
    to_camel_case = lambda snake_str: "".join(
        x.capitalize() for x in snake_str.lower().split("_")
    )
    dataclass_name = f"{to_camel_case(task_name)}Inputs"
    return make_dataclass(dataclass_name, fields)


@dataclass
class TaskUserFunction(UserFunction):
    """A UserFunction that represents an agent task, not a regular function."""

    # Required fields for task execution (with defaults to satisfy dataclass ordering)
    task_agent_fingerprint: str = ""  # Agent that will execute the task
    task_agent_name: str = ""  # Agent name (for fallback resolution)
    task_docstring: str = ""  # Task instructions
    task_return_type: type = object  # Expected return type
    # Allow network access for task calls (needed for LLM calls in sub-agent tasks)
    network_access: bool = True

    def execute(
        self, args: list, kwargs: dict, source_code: str | None, parent_evaluator=None
    ):
        """Override execute to run task loop instead of function body via agent.run_task."""
        # Resolve the task-executing agent, with fallback by name
        task_agent = None
        try:
            task_agent = resolve_agent(self.task_agent_fingerprint)
        except RuntimeError:
            # Fingerprint not found - try fallback by agent name
            if self.task_agent_name:
                task_agent = get_agent_by_name(self.task_agent_name)
            if task_agent:
                warnings.warn(
                    f"Task function '{self.name}' was created by agent with fingerprint "
                    f"'{self.task_agent_fingerprint[:8]}...' which is no longer registered. "
                    f"Falling back to agent '{task_agent.name}' by name. "
                    f"This may happen after config changes (primer, registrations).",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    f"Cannot execute task function '{self.name}': "
                    f"No agent found with fingerprint '{self.task_agent_fingerprint[:8]}...' "
                    f"or name '{self.task_agent_name}'."
                )

        # Agent.run_task expects the wrapper callable which embeds loop invocation
        # We synthesize an adapter that validates args and calls the loop
        def _task_wrapper_adapter(*_args, **_kwargs):
            # Extract state and on_event if present (run_task will set them already)
            _kwargs.pop("state", None)
            _kwargs.pop("on_event", None)

            inputs_dataclass = create_inputs_dataclass_from_ast_args(
                self.name, self.args, use_generic_types=True
            )
            inputs_instance = self._create_inputs_instance(
                list(_args), _kwargs, inputs_dataclass
            )

            from agex.agent import Agent

            if isinstance(task_agent, Agent):
                return task_agent._run_task_loop(
                    task_name=self.name,
                    docstring=self.task_docstring,
                    inputs_dataclass=inputs_dataclass,
                    inputs_instance=inputs_instance,
                    return_type=self.task_return_type,
                    state=_kwargs.get("state"),
                    on_event=_kwargs.get("on_event"),
                )
            raise RuntimeError(
                f"Task agent {self.task_agent_fingerprint} is not a valid Agent instance"
            )

        # Delegate through agent.run_task for consistent state management
        session = "default"
        if parent_evaluator is not None:
            session = parent_evaluator.session
        else:
            session = kwargs.pop("session", "default")

        on_event = None
        if parent_evaluator is not None:
            on_event = getattr(parent_evaluator, "on_event", None)
        else:
            on_event = kwargs.pop("on_event", None)

        on_token = None
        if parent_evaluator is not None:
            on_token = getattr(parent_evaluator, "on_token", None)
        else:
            on_token = kwargs.pop("on_token", None)

        return task_agent.run_task(
            _task_wrapper_adapter,
            args,
            kwargs,
            session,
            on_event=on_event,
            on_token=on_token,
        )

    def _create_inputs_instance(self, args: list, kwargs: dict, inputs_dataclass: type):
        """Create an instance of the inputs dataclass with the provided arguments."""
        if not args and not kwargs:
            return None if not self.args.args else inputs_dataclass()

        # Bind arguments to parameter names
        param_names = [arg.arg for arg in self.args.args]
        bound_args = {}

        # Handle positional arguments
        for i, value in enumerate(args):
            if i < len(param_names):
                bound_args[param_names[i]] = value

        # Handle keyword arguments
        for name, value in kwargs.items():
            if name in param_names:
                bound_args[name] = value

        return inputs_dataclass(**bound_args) if bound_args else None


class TaskProxy:
    """
    Execution proxy for dual-decorated task callables (wrappers created by @agent.task).

    This moves the execution-time logic (state namespacing, event propagation,
    timeout accounting) out of the evaluator and into a dedicated class.
    """

    def __init__(self, evaluator: "BaseEvaluator", task_callable: Any):
        from agex.eval.base import (
            BaseEvaluator as _BaseEvaluator,
        )

        if not isinstance(evaluator, _BaseEvaluator):
            raise TypeError("TaskProxy requires a BaseEvaluator instance")
        self.evaluator = evaluator
        self.task_callable = task_callable

    def execute(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        """
        Execute a sub-agent task, properly accounting for execution time.

        For sync tasks: Time is measured and added before returning.
        For async tasks: Result is wrapped to measure time after await completes.
        """
        # Use session from evaluator for sub-agent to inherit
        session = self.evaluator.session

        sub_agent_start = time.time()
        try:
            # Delegate execution to the agent with inherited session
            agent = self.evaluator.agent
            result = agent.run_task(
                self.task_callable,
                args,
                kwargs,
                session=session,
                on_event=getattr(self.evaluator, "on_event", None),
                on_token=getattr(self.evaluator, "on_token", None),
            )

            # Handle async results: wrap to capture timing *after* await completes
            if inspect.isawaitable(result):

                async def _timed_wrapper(awaitable):
                    try:
                        return await awaitable
                    finally:
                        sub_agent_duration = time.time() - sub_agent_start
                        self._safe_add_sub_agent_time(sub_agent_duration)

                return _timed_wrapper(result)

            # Sync success: account for time before returning
            sub_agent_duration = time.time() - sub_agent_start
            self._safe_add_sub_agent_time(sub_agent_duration)
            return result

        except Exception:
            # Sync error: still account for time spent
            sub_agent_duration = time.time() - sub_agent_start
            self._safe_add_sub_agent_time(sub_agent_duration)
            raise

    def _safe_add_sub_agent_time(self, duration: float) -> None:
        """Safely add sub-agent time to the evaluator, ignoring errors."""
        try:
            self.evaluator.add_sub_agent_time(duration)
        except Exception:
            pass


class FunctionEvaluator(BaseEvaluator):
    """A mixin for evaluating function definition and return nodes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles function definitions."""
        free_vars = get_free_variables(node)

        # Exclude registered functions from closure capture - they should be resolved via policy
        # But only if they're not already bound in the current state (i.e., they're not local variables)
        main_ns = self.agent._policy.namespaces.get("__main__")
        if main_ns:
            registered_fns = set(main_ns.fn_objects.keys())
            registered_classes = set(main_ns.classes.keys())
            # Only exclude if the variable isn't already defined in current scope
            free_vars = free_vars - {
                name for name in registered_fns if name not in self.state
            }
            free_vars = free_vars - {
                name for name in registered_classes if name not in self.state
            }

        closure = LiveClosureState(self.state, free_vars)

        source_text = None
        if self.source_code:
            try:
                source_text = ast.get_source_segment(self.source_code, node)
            except (IndexError, ValueError):
                # Source extraction can fail in rehydrated contexts
                # where line numbers don't align properly
                source_text = None

        # Extract docstring from function body (first statement if it's a string literal)
        docstring = None
        if node.body:
            first_stmt = node.body[0]
            if isinstance(first_stmt, ast.Expr):
                # Check for string constant (Python 3.8+)
                if isinstance(first_stmt.value, ast.Constant) and isinstance(
                    first_stmt.value.value, str
                ):
                    docstring = first_stmt.value.value

        # Evaluate default argument values NOW (at definition time)
        # This matches Python semantics where default values are captured when
        # the function is defined, not when it's called.
        evaluated_defaults = None
        if node.args.defaults:
            evaluated_defaults = [self.visit(d) for d in node.args.defaults]

        evaluated_kw_defaults = None
        if node.args.kw_defaults:
            evaluated_kw_defaults = [
                self.visit(d) if d is not None else None for d in node.args.kw_defaults
            ]

        func = UserFunction(
            name=node.name,
            args=node.args,
            body=node.body,
            closure_state=closure,
            source_text=source_text,
            agent_fingerprint=self.agent.fingerprint,
            agent_name=self.agent.name,
            session=self.session,
            evaluated_defaults=evaluated_defaults,
            evaluated_kw_defaults=evaluated_kw_defaults,
        )
        # Set the docstring as a Python-compatible attribute
        func.__doc__ = docstring

        # Apply decorators in reverse order (inner -> outer)
        # We start with the UserFunction 'func' and wrap it layer by layer
        decorated_func = func
        for decorator_node in reversed(node.decorator_list):
            decorator = self.visit(decorator_node)
            if not callable(decorator):
                from agex.eval.error import EvalError

                raise EvalError(
                    f"Decorator '{ast.unparse(decorator_node)}' is not callable",
                    decorator_node,
                )

            try:
                decorated_func = decorator(decorated_func)
            except Exception as e:
                from agex.eval.error import EvalError

                raise EvalError(
                    f"Error applying decorator '{ast.unparse(decorator_node)}': {e}",
                    decorator_node,
                )

        self.state.set(node.name, decorated_func)

        # Track user function names for system prompts (shadow set)
        # We use a shadow set to avoid iterating the entire state to find functions.
        current_names = self.state.get("__sys_user_fn_names__", set())
        if node.name not in current_names:
            # Create new set to ensure we trigger state update
            new_names = current_names | {node.name}
            self.state.set("__sys_user_fn_names__", new_names)

    def visit_Lambda(self, node: ast.Lambda) -> UserFunction:
        """Handles lambda expressions."""
        free_vars = get_free_variables(node)

        # Exclude registered functions from closure capture - they should be resolved via policy
        # But only if they're not already bound in the current state (i.e., they're not local variables)
        main_ns = self.agent._policy.namespaces.get("__main__")
        if main_ns:
            registered_fns = set(main_ns.fn_objects.keys())
            registered_classes = set(main_ns.classes.keys())
            # Only exclude if the variable isn't already defined in current scope
            free_vars = free_vars - {
                name for name in registered_fns if name not in self.state
            }
            free_vars = free_vars - {
                name for name in registered_classes if name not in self.state
            }

        closure = LiveClosureState(self.state, free_vars)

        source_text = None
        if self.source_code:
            try:
                source_text = ast.get_source_segment(self.source_code, node)
            except (IndexError, ValueError):
                # Source extraction can fail in rehydrated contexts
                # where line numbers don't align properly
                source_text = None

        # Evaluate default argument values NOW (at definition time)
        # This is critical for correct semantics: lambda f=fruit: ... should
        # capture the VALUE of fruit when the lambda is created, not re-evaluate
        # the name 'fruit' when the lambda is called.
        evaluated_defaults = None
        if node.args.defaults:
            evaluated_defaults = [self.visit(d) for d in node.args.defaults]

        evaluated_kw_defaults = None
        if node.args.kw_defaults:
            evaluated_kw_defaults = [
                self.visit(d) if d is not None else None for d in node.args.kw_defaults
            ]

        return UserFunction(
            name="<lambda>",
            args=node.args,
            body=[ast.Return(value=node.body)],  # Lambdas are a single expression
            closure_state=closure,
            source_text=source_text,
            agent_fingerprint=self.agent.fingerprint,
            agent_name=self.agent.name,
            session=self.session,
            evaluated_defaults=evaluated_defaults,
            evaluated_kw_defaults=evaluated_kw_defaults,
        )

    def visit_Return(self, node: ast.Return) -> None:
        """Handles return statements."""
        value = self.visit(node.value) if node.value else None
        raise _ReturnException(value, node)
