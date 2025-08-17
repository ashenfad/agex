import ast
from dataclasses import dataclass, make_dataclass
from typing import Any, Callable

from agex.agent.base import resolve_agent
from agex.state import Live

from ..state import State
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
    closure_state: State  # A *reference* to the state where the function was defined.
    source_text: str | None = None
    agent_fingerprint: str | None = (
        None  # Fingerprint of the agent this function was defined in
    )

    # Ensure hashability for use in libraries that cache by callable (e.g., pandas.apply)
    def __hash__(self) -> int:  # type: ignore[override]
        # Identity-based hash keeps semantics simple and avoids mutable field issues
        return hash(id(self))

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        return self is other

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
        from agex.eval.arguments import bind_arguments
        from agex.eval.core import Evaluator

        exec_state = Scoped(self.closure_state)

        if not self.agent_fingerprint:
            raise RuntimeError("Cannot execute function without an agent context.")

        # Resolve agent from fingerprint
        agent = resolve_agent(self.agent_fingerprint)

        # Create evaluator with timeout context from parent if available
        if parent_evaluator is not None:
            # Inherit timeout context from parent evaluator
            evaluator = Evaluator(
                agent=agent,
                state=exec_state,
                source_code=source_code,
                timeout_seconds=parent_evaluator._timeout_seconds,
                start_time=parent_evaluator._start_time,
                sub_agent_time=parent_evaluator._sub_agent_time,
            )
        else:
            # Fresh timeout budget (for direct calls)
            evaluator = Evaluator(
                agent=agent,
                state=exec_state,
                source_code=source_code,
                timeout_seconds=agent.timeout_seconds,
            )
        bound_args = bind_arguments(
            self.name, self.args, args, kwargs, eval_fn=evaluator.visit
        )
        for name, value in bound_args.items():
            exec_state.set(name, value)

        try:
            for node in self.body:
                evaluator.visit(node)
            return None
        except _ReturnException as e:
            return e.value


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
    task_docstring: str = ""  # Task instructions
    task_return_type: type = object  # Expected return type

    def execute(
        self, args: list, kwargs: dict, source_code: str | None, parent_evaluator=None
    ):
        """Override execute to run task loop instead of function body via agent.run_task."""
        # Resolve the task-executing agent
        task_agent = resolve_agent(self.task_agent_fingerprint)

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
        parent_state = None
        if parent_evaluator is not None:
            parent_state = parent_evaluator.state
        else:
            # When executed directly, no parent evaluator means no parent state; pass through provided state
            parent_state = kwargs.pop("state", Live())

        on_event = None
        if parent_evaluator is not None:
            on_event = getattr(parent_evaluator, "on_event", None)
        else:
            on_event = kwargs.pop("on_event", None)

        return task_agent.run_task(
            _task_wrapper_adapter,
            args,
            kwargs,
            parent_state,
            on_event=on_event,
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
        # Measure sub-agent call time to deduct from parent timeout; execution is delegated to the agent
        import time

        sub_agent_start = time.time()
        try:
            # Determine parent state
            from ..state import Live, Versioned
            from ..state import Namespaced as NamespacedState

            if isinstance(self.evaluator.state, (Versioned, NamespacedState, Live)):
                parent_state = self.evaluator.state
            else:
                parent_state = self.evaluator.state.base_store

            # Delegate execution and state management to the agent
            agent = self.evaluator.agent
            return agent.run_task(
                self.task_callable,
                args,
                kwargs,
                parent_state,
                on_event=getattr(self.evaluator, "on_event", None),
            )
        finally:
            sub_agent_duration = time.time() - sub_agent_start
            # Inform evaluator so it can adjust time budget
            try:
                self.evaluator.add_sub_agent_time(sub_agent_duration)
            except Exception:
                pass


class FunctionEvaluator(BaseEvaluator):
    """A mixin for evaluating function definition and return nodes."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Handles function definitions."""
        free_vars = get_free_variables(node)
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

        func = UserFunction(
            name=node.name,
            args=node.args,
            body=node.body,
            closure_state=closure,
            source_text=source_text,
            agent_fingerprint=self.agent.fingerprint,
        )
        # Set the docstring as a Python-compatible attribute
        func.__doc__ = docstring
        self.state.set(node.name, func)

    def visit_Lambda(self, node: ast.Lambda) -> UserFunction:
        """Handles lambda expressions."""
        free_vars = get_free_variables(node)
        closure = LiveClosureState(self.state, free_vars)

        source_text = None
        if self.source_code:
            try:
                source_text = ast.get_source_segment(self.source_code, node)
            except (IndexError, ValueError):
                # Source extraction can fail in rehydrated contexts
                # where line numbers don't align properly
                source_text = None

        return UserFunction(
            name="<lambda>",
            args=node.args,
            body=[ast.Return(value=node.body)],  # Lambdas are a single expression
            closure_state=closure,
            source_text=source_text,
            agent_fingerprint=self.agent.fingerprint,
        )

    def visit_Return(self, node: ast.Return) -> None:
        """Handles return statements."""
        value = self.visit(node.value) if node.value else None
        raise _ReturnException(value, node)
