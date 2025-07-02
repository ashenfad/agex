import ast
from dataclasses import dataclass, make_dataclass
from typing import Any, Callable

from agex.agent.base import resolve_agent

from ..state import State
from ..state.closure import LiveClosureState
from ..state.scoped import Scoped
from .analysis import get_free_variables
from .base import BaseEvaluator


class _ReturnException(Exception):
    """Internal exception to signal a return statement, carrying the return value."""

    def __init__(self, value: Any):
        self.value = value


@dataclass
class NativeFunction:
    """Represents a native Python function available in the Tic environment."""

    name: str
    fn: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Directly call the wrapped native function.
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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self.agent_fingerprint:
            raise RuntimeError(
                "UserFunction cannot be called directly without an Agent context."
            )
        # No source code available from fingerprint
        return self.execute(list(args), kwargs, None)

    def execute(self, args: list, kwargs: dict, source_code: str | None):
        """Execute the function with a new evaluator."""
        from agex.eval.arguments import bind_arguments
        from agex.eval.core import Evaluator

        exec_state = Scoped(self.closure_state)

        if not self.agent_fingerprint:
            raise RuntimeError("Cannot execute function without an agent context.")

        # Resolve agent from fingerprint
        agent = resolve_agent(self.agent_fingerprint)

        evaluator = Evaluator(
            agent=agent,
            state=exec_state,
            source_code=source_code,
            # Functions inherit the agent's timeout
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

    def execute(self, args: list, kwargs: dict, source_code: str | None):
        """Override execute to run task loop instead of function body."""
        # Resolve the task-executing agent
        task_agent = resolve_agent(self.task_agent_fingerprint)

        # Extract state parameter (injected by task calling convention)
        state = kwargs.pop("state", None)

        # Create generic inputs dataclass using shared utility
        inputs_dataclass = create_inputs_dataclass_from_ast_args(
            self.name, self.args, use_generic_types=True
        )
        inputs_instance = self._create_inputs_instance(args, kwargs, inputs_dataclass)

        # Trigger the task loop
        from agex.agent import Agent

        if isinstance(task_agent, Agent):
            return task_agent._run_task_loop(
                task_name=self.name,
                docstring=self.task_docstring,
                inputs_dataclass=inputs_dataclass,
                inputs_instance=inputs_instance,
                return_type=self.task_return_type,
                state=state,
            )
        else:
            raise RuntimeError(
                f"Task agent {self.task_agent_fingerprint} is not a valid Agent instance"
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

        func = UserFunction(
            name=node.name,
            args=node.args,
            body=node.body,
            closure_state=closure,
            source_text=source_text,
            agent_fingerprint=self.agent.fingerprint,
        )
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
        raise _ReturnException(value)
