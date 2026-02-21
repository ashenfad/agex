import ast
import inspect
import warnings
from dataclasses import dataclass, make_dataclass
from typing import Any, Callable

from agex.agent.base import get_agent_by_name, resolve_agent


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
    """Represents a user-defined function from the old AST-walking eval engine.

    Retained for backward compatibility with serialized state and isinstance
    checks in the agent layer. New sandbox-defined functions use SbFunction.
    """

    name: str
    args: ast.arguments
    body: list[ast.stmt]
    closure_state: Any  # A *reference* to the state where the function was defined.
    source_text: str | None = None
    agent_fingerprint: str | None = None
    agent_name: str | None = None
    session: str = "default"
    evaluated_defaults: list[Any] | None = None
    evaluated_kw_defaults: list[Any] | None = None

    # Ensure hashability for use in libraries that cache by callable (e.g., pandas.apply)
    def __hash__(self) -> int:  # type: ignore[override]
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
        raise RuntimeError(
            f"UserFunction '{self.name}' cannot be called: the old AST-walking "
            f"eval engine has been removed. Sandbox-defined functions now use "
            f"SbFunction via sblite."
        )


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
    """A UserFunction that represents an agent task, not a regular function.

    Retained for backward compatibility with serialized state and isinstance
    checks. New task calls use _wrap_sub_agent_task in the bridge layer.
    """

    task_agent_fingerprint: str = ""
    task_agent_name: str = ""
    task_docstring: str = ""
    task_return_type: type = object
    network_access: bool = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the sub-agent task via agent.run_task."""
        task_agent = None
        try:
            task_agent = resolve_agent(self.task_agent_fingerprint)
        except RuntimeError:
            if self.task_agent_name:
                task_agent = get_agent_by_name(self.task_agent_name)
            if task_agent:
                warnings.warn(
                    f"Task function '{self.name}' was created by agent with fingerprint "
                    f"'{self.task_agent_fingerprint[:8]}...' which is no longer registered. "
                    f"Falling back to agent '{task_agent.name}' by name.",
                    stacklevel=2,
                )
            else:
                raise RuntimeError(
                    f"Cannot execute task function '{self.name}': "
                    f"No agent found with fingerprint '{self.task_agent_fingerprint[:8]}...' "
                    f"or name '{self.task_agent_name}'."
                )

        inputs_dataclass = create_inputs_dataclass_from_ast_args(
            self.name, self.args, use_generic_types=True
        )
        inputs_instance = self._create_inputs_instance(
            list(args), kwargs, inputs_dataclass
        )

        def _task_wrapper_adapter(*_args, **_kwargs):
            _kwargs.pop("state", None)
            _kwargs.pop("on_event", None)

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

        session = kwargs.pop("session", "default")
        on_event = kwargs.pop("on_event", None)
        on_token = kwargs.pop("on_token", None)

        return task_agent.run_task(
            _task_wrapper_adapter,
            list(args),
            kwargs,
            session=session,
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
