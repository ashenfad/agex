"""
Task decorator mixin for Agent class.

This module provides the TaskMixin that handles the @agent.task decorator
which wraps functions to become agent tasks.
"""

import inspect
from dataclasses import make_dataclass
from typing import Any, Callable

from agex.agent.base import BaseAgent
from agex.agent.loop import TaskLoopMixin
from agex.agent.utils import is_function_body_empty
from agex.eval.validation import validate_with_sampling

# Global registry for dynamically created input dataclasses
# This allows pickle to find them by module.classname lookup
_DYNAMIC_DATACLASS_REGISTRY: dict[str, type] = {}


def clear_dynamic_dataclass_registry() -> None:
    """Clear the dynamic dataclass registry. Useful for testing or memory management."""
    global _DYNAMIC_DATACLASS_REGISTRY
    # Remove from module globals
    for class_name in list(_DYNAMIC_DATACLASS_REGISTRY.keys()):
        globals().pop(class_name, None)
    # Clear the registry
    _DYNAMIC_DATACLASS_REGISTRY.clear()


class TaskMixin(TaskLoopMixin, BaseAgent):
    def run_task(
        self,
        task_callable: Callable,
        args: list,
        kwargs: dict,
        parent_state,
        on_event: Callable[[Any], None] | None = None,
    ) -> Any:
        """
        Execute a task callable within a namespaced child context of the parent state.

        This centralizes sub-task state management and event propagation.

        Args:
            task_callable: The callable produced by @agent.task
            args: Positional arguments to pass to the task
            kwargs: Keyword arguments to pass to the task
            parent_state: The parent's execution state (Versioned/Namespaced/Live)
            on_event: Optional event handler to propagate

        Returns:
            The task result produced by the task loop
        """
        from agex.state import Namespaced

        namespace = getattr(task_callable, "__agex_task_namespace__", self.name)
        child_state = Namespaced(parent_state, namespace)

        # Prepare kwargs safely
        call_kwargs = dict(kwargs) if kwargs is not None else {}
        call_kwargs["state"] = child_state
        if on_event is not None:
            call_kwargs["on_event"] = on_event

        return task_callable(*args, **call_kwargs)

    def task(
        self,
        primer_or_func=None,
        /,
        *,
        primer: str | None = None,
        setup: str | None = None,
    ) -> Callable:
        """
        Decorator to mark a function as an agent task.

        The decorated function must have an empty body (only pass, docstrings, comments).
        The decorator replaces the function with one that triggers the agent's task loop.

        Usage:
            # Naked decorator - uses docstring for agent instructions
            @agent.task
            def my_function():
                '''Clear instructions for both agent and caller.'''
                pass

            # Parameterized with no args - same as naked
            @agent.task()
            def my_function():
                '''Clear instructions for both agent and caller.'''
                pass

            # Parameterized with primer - primer for agent, docstring for caller
            @agent.task("Detailed agent implementation instructions")
            def my_function():
                '''Public API documentation for callers.'''
                pass

            # With setup code for context discovery
            @agent.task(setup="schema = db.execute('PRAGMA table_info(sales)').fetchall()")
            def query_database():
                '''Query the database and return results.'''
                pass

        Args:
            primer_or_func: Either the primer string or the function being decorated
            primer: Keyword-only primer argument (alternative to positional)
            setup: Optional code string to execute before the task for context discovery.
                   This runs automatically and doesn't count against iteration limits.

        Returns:
            Either the decorated function (naked) or a decorator function (parameterized)
        """

        def decorator(func: Callable) -> Callable:
            # Check if this is a UserFunction (agent creating task from another agent's function)
            from agex.eval.functions import TaskUserFunction, UserFunction

            if isinstance(func, UserFunction):
                # Special case: creating task from existing UserFunction
                # Determine the effective primer
                effective_primer = primer
                if effective_primer is None and not callable(primer_or_func):
                    effective_primer = primer_or_func

                # Use the UserFunction's docstring
                if effective_primer is None:
                    effective_primer = (
                        func.__doc__ or "Execute the user-defined function as a task."
                    )

                # Create TaskUserFunction
                return TaskUserFunction(
                    # Copy UserFunction metadata
                    name=func.name,
                    args=func.args,
                    body=func.body,
                    closure_state=func.closure_state,
                    source_text=func.source_text,
                    agent_fingerprint=func.agent_fingerprint,
                    # Add task-specific metadata
                    task_agent_fingerprint=self.fingerprint,
                    task_docstring=effective_primer,
                    task_return_type=object,  # Generic type since UserFunction loses type hints
                )
            else:
                # Normal case: real function definition
                self._validate_task_decorator(func)

                # Determine the effective primer. The keyword 'primer' takes highest precedence.
                # If not provided, check if a positional primer was passed (in which case
                # primer_or_func will be a string, not the function being decorated).
                effective_primer = primer
                if effective_primer is None and not callable(primer_or_func):
                    effective_primer = primer_or_func

                return self._create_task_wrapper(
                    func, primer=effective_primer, setup=setup
                )

        # If the decorator is used without parentheses (@agent.task), the function
        # is passed directly as primer_or_func. In this case, we call the decorator
        # immediately with the function.
        if callable(primer_or_func):
            return decorator(primer_or_func)

        # If the decorator is used with parentheses (@agent.task(...)), we return
        # the decorator itself. Python will then call it with the decorated function.
        return decorator

    def _validate_task_decorator(self, func: Callable) -> None:
        """Validate that task decorator is being used correctly."""
        # 1. Prevent multiple task decorators (no multi-agent tasks)
        if hasattr(func, "__agex_task_namespace__"):
            existing_namespace = func.__agex_task_namespace__
            raise ValueError(
                f"Function '{func.__name__}' already has a task decorator (namespace: '{existing_namespace}'). "
                f"Multi-agent tasks are not supported."
            )

        # 2. Prevent wrong decorator order (fn must be outer)
        if hasattr(func, "__is_agent_fn__"):
            raise ValueError(
                f"Invalid decorator order on '{func.__name__}'. "
                f"@agent.fn() must be applied AFTER @agent.task(), not before.\n"
                f"Correct order:\n"
                f"@agent.fn()\n"
                f"@agent.task('...')\n"
                f"def {func.__name__}(): ..."
            )

    def _create_task_wrapper(
        self, func: Callable, primer: str | None, setup: str | None = None
    ) -> Callable:
        """
        Creates the actual task wrapper function.

        Args:
            func: The original function to wrap
            primer: Agent instructions for implementing the task (None to use docstring)

        Returns:
            The wrapped function
        """
        # Validate that the function body is empty
        if not is_function_body_empty(func):
            raise ValueError(
                f"Function '{func.__name__}' decorated with @task must have an empty body. "
                "The agent will provide the implementation."
            )

        # Capture original function metadata
        original_sig = inspect.signature(func)
        return_type = original_sig.return_annotation
        task_name = func.__name__

        # Determine effective agent instructions
        if primer is not None:
            # Use provided primer for agent instructions
            effective_docstring = primer
        else:
            # Fall back to function docstring
            if func.__doc__ is None or func.__doc__.strip() == "":
                raise ValueError(
                    f"Function '{func.__name__}' decorated with @task must have either "
                    "a primer argument or a non-empty docstring to provide agent instructions."
                )
            effective_docstring = func.__doc__.strip()

        # Create dynamic dataclass for inputs
        inputs_dataclass = self._create_inputs_dataclass(task_name, original_sig)

        # Create new signature with added state parameter
        # Insert state parameter before **kwargs if it exists, otherwise append at end
        new_params = list(original_sig.parameters.values())
        state_param = inspect.Parameter(
            "state",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Versioned | Live | None",
        )
        on_event_param = inspect.Parameter(
            "on_event",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Callable[[BaseEvent], None] | None",
        )

        # Find if there's a **kwargs parameter (VAR_KEYWORD)
        var_keyword_index = None
        for i, param in enumerate(new_params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                var_keyword_index = i
                break

        if var_keyword_index is not None:
            # Insert parameters before **kwargs
            new_params.insert(var_keyword_index, on_event_param)
            new_params.insert(var_keyword_index, state_param)
        else:
            # No **kwargs, append at end
            new_params.append(state_param)
            new_params.append(on_event_param)

        new_sig = original_sig.replace(parameters=new_params)

        # Create a custom callable class with proper __repr__
        class TaskWrapper:
            def __init__(self, task_func, stream_func, agent_name, task_name):
                self._task_func = task_func
                self._stream_func = stream_func
                self._agent_name = agent_name
                self._task_name = task_name

                # Copy function attributes
                self.__name__ = func.__name__
                self.__doc__ = func.__doc__
                self.__annotations__ = func.__annotations__.copy()
                self.__annotations__["state"] = "Versioned | Live | None"
                self.__annotations__["on_event"] = "Callable[[BaseEvent], None] | None"
                self.__signature__ = new_sig

                # Set namespace for dual-decorator pattern
                namespace = self._agent_name
                self.__agex_task_namespace__ = namespace

            def __call__(self, *args, **kwargs):
                return self._task_func(*args, **kwargs)

            def __repr__(self):
                return f"<agex.task {self._agent_name}/{self._task_name} at {hex(id(self))}>"

            @property
            def stream(self):
                return self._stream_func

        # Create the actual task function
        def task_wrapper(*args, **kwargs):
            # Bind to the new signature that includes the 'state' and 'on_event' parameters
            bound_args = new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Pop the state and on_event arguments, they are handled separately
            state = bound_args.arguments.pop("state", None)
            on_event = bound_args.arguments.pop("on_event", None)

            # Create inputs dataclass instance with pass-by-value semantics
            inputs_instance = None
            if bound_args.arguments:
                validated_args = {}
                for name, value in bound_args.arguments.items():
                    annotation = original_sig.parameters[name].annotation
                    if annotation == inspect.Parameter.empty:
                        annotation = Any  # Default to Any if no type hint
                    try:
                        validated_value = validate_with_sampling(value, annotation)
                        validated_args[name] = validated_value
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for argument '{name}':\n{e}"
                        ) from e
                inputs_instance = inputs_dataclass(**validated_args)

            # Call the task loop
            return self._run_task_loop(
                task_name=task_name,
                docstring=effective_docstring,
                inputs_dataclass=inputs_dataclass,
                inputs_instance=inputs_instance,
                return_type=return_type,
                state=state,
                on_event=on_event,
                setup=setup,
            )

        def stream(*args, **kwargs):
            """Stream events in real-time during task execution."""
            # Same parameter processing as regular task execution
            bound_args = new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Pop the state and on_event arguments, they are handled separately
            state = bound_args.arguments.pop("state", None)
            user_on_event = bound_args.arguments.pop("on_event", None)

            # Create inputs dataclass instance with pass-by-value semantics
            inputs_instance = None
            if bound_args.arguments:
                validated_args = {}
                for name, value in bound_args.arguments.items():
                    annotation = original_sig.parameters[name].annotation
                    if annotation == inspect.Parameter.empty:
                        annotation = Any  # Default to Any if no type hint
                    try:
                        validated_value = validate_with_sampling(value, annotation)
                        validated_args[name] = validated_value
                    except Exception as e:
                        raise ValueError(
                            f"Validation failed for argument '{name}':\n{e}"
                        ) from e
                inputs_instance = inputs_dataclass(**validated_args)

            # Implement real-time hierarchical streaming using a worker thread and queue
            from queue import Queue
            from threading import Event as _ThreadEvent
            from threading import Thread

            _SENTINEL = object()
            _queue: Queue = Queue()
            _done = _ThreadEvent()

            def _handler(ev):
                # Enqueue every event and optionally forward to user handler
                try:
                    _queue.put(ev)
                finally:
                    if user_on_event is not None:
                        try:
                            user_on_event(ev)
                        except Exception:
                            # Swallow user handler errors to avoid breaking streaming
                            pass

            def _run_task():
                try:
                    # Execute standard (non-streaming) loop; events flow via _handler
                    self._run_task_loop(
                        task_name=task_name,
                        docstring=effective_docstring,
                        inputs_dataclass=inputs_dataclass,
                        inputs_instance=inputs_instance,
                        return_type=return_type,
                        state=state,
                        on_event=_handler,
                        setup=setup,
                    )
                except BaseException as e:
                    # Emit the exception into the queue so the consumer can re-raise
                    try:
                        _queue.put(e)
                    finally:
                        pass
                finally:
                    try:
                        _queue.put(_SENTINEL)
                    finally:
                        _done.set()

            _thread = Thread(target=_run_task, daemon=True)
            _thread.start()

            def _event_generator():
                try:
                    while True:
                        ev = _queue.get()
                        # If the worker enqueued an exception, re-raise it to match test expectations
                        if isinstance(ev, BaseException):
                            raise ev
                        if ev is _SENTINEL:
                            break
                        yield ev
                finally:
                    if not _done.is_set():
                        _done.wait(timeout=1.0)
                    _thread.join(timeout=1.0)

            return _event_generator()

        # Create the custom wrapper with proper __repr__
        agent_name = self.name if self.name is not None else self.__class__.__name__
        wrapper = TaskWrapper(task_wrapper, stream, agent_name, task_name)

        return wrapper

    def _create_inputs_dataclass(self, task_name: str, signature: inspect.Signature):
        """
        Create a dynamic dataclass for the task inputs.

        Args:
            task_name: Name of the task function
            signature: Function signature to extract parameters from

        Returns:
            Dynamically created dataclass type
        """
        if not signature.parameters:
            # No inputs - return a simple empty dataclass
            return make_dataclass(f"{task_name.title()}Inputs", [])

        # Build field specifications for make_dataclass
        fields = []
        for param_name, param in signature.parameters.items():
            # Get type annotation, default to Any if not specified
            param_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else Any
            )

            # Handle default values
            if param.default != inspect.Parameter.empty:
                # Has default value
                fields.append((param_name, param_type, param.default))
            else:
                # Required parameter
                fields.append((param_name, param_type))

        # Create the dataclass
        to_camel_case = lambda snake_str: "".join(
            x.capitalize() for x in snake_str.lower().split("_")
        )
        dataclass_name = f"{to_camel_case(task_name)}Inputs"
        inputs_dataclass = make_dataclass(dataclass_name, fields)

        # Make the dataclass pickleable by registering it in module globals
        # This allows pickle to find it via module.classname lookup
        inputs_dataclass.__module__ = __name__  # Set to this module
        _DYNAMIC_DATACLASS_REGISTRY[dataclass_name] = inputs_dataclass
        globals()[dataclass_name] = inputs_dataclass  # Make it findable by pickle

        # Register the dataclass with the agent for sandbox access
        if hasattr(self, "cls"):
            self.cls(inputs_dataclass, constructable=False)  # type: ignore

        return inputs_dataclass
