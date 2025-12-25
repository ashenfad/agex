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
from agex.host import Local

# Global registry for dynamically created input dataclasses
# This allows pickle to find them by module.classname lookup
_DYNAMIC_DATACLASS_REGISTRY: dict[str, type] = {}


def _register_dataclass_for_pickling(dc_class: type, name: str) -> None:
    """
    Register a dynamic dataclass so cloudpickle can serialize it by value.

    This is called when creating dynamic dataclasses to ensure they can
    be pickled and sent to remote servers that don't have the class defined.
    """

    # Set the module to this one so pickle can find it
    dc_class.__module__ = __name__
    _DYNAMIC_DATACLASS_REGISTRY[name] = dc_class
    globals()[name] = dc_class

    try:
        import sys

        import cloudpickle

        # Register this module to be pickled by value, not by reference
        this_module = sys.modules[__name__]
        cloudpickle.register_pickle_by_value(this_module)
    except (ImportError, AttributeError, ValueError):
        pass  # cloudpickle not available or doesn't support this API


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
        on_token: Callable[[Any], None] | None = None,
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

        # Rehydrate sub-agent if needed (may have llm=None after deserialization)
        # This happens when cloudpickle deserializes nested agents in closures
        sub_agent = getattr(task_callable, "__agex_agent__", None)
        if sub_agent is not None:
            # Rehydrate LLM from serialized config if missing
            if sub_agent.llm is None:
                if hasattr(sub_agent, "_llm_config") and sub_agent._llm_config:
                    from agex.llm import LLM

                    sub_agent.llm = LLM.from_config(sub_agent._llm_config)

            # Re-register in global registry so UserFunctions can resolve it
            # (fingerprint is None after deserialization until registered)
            if sub_agent.fingerprint is None:
                sub_agent._update_fingerprint()

        # Prepare kwargs safely
        call_kwargs = dict(kwargs) if kwargs is not None else {}
        call_kwargs["_parent_state"] = child_state
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        return task_callable(*args, **call_kwargs)

    def task(
        self,
        primer_or_func=None,
        /,
        *,
        primer: str | None = None,
        setup: str | None = None,
        on_conflict: str = "retry",
        max_conflict_retries: int = 3,
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
            on_conflict: How to handle concurrency conflicts when merging Versioned state.
                'retry' (default) - Automatically retry the task with fresh state
                'abandon' - Silently abandon the work (commits become orphans for GC)
            max_conflict_retries: Maximum number of retry attempts (default: 3)

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
                wrapper = TaskUserFunction(
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
                # Attach agent instance for @remote decorator access
                wrapper.__agex_agent__ = self  # type: ignore
                return wrapper
            else:
                # Normal case: real function definition
                self._validate_task_decorator(func)

                # Determine the effective primer. The keyword 'primer' takes highest precedence.
                # If not provided, check if a positional primer was passed (in which case
                # primer_or_func will be a string, not the function being decorated).
                effective_primer = primer
                if effective_primer is None and not callable(primer_or_func):
                    effective_primer = primer_or_func

                wrapper = self._create_task_wrapper(
                    func,
                    primer=effective_primer,
                    setup=setup,
                    on_conflict=on_conflict,
                    max_conflict_retries=max_conflict_retries,
                )

                # Register the task so it can be found remotely
                if hasattr(self, "_tasks"):
                    self._tasks[func.__name__] = wrapper

                return wrapper

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
        self,
        func: Callable,
        primer: str | None,
        setup: str | None = None,
        on_conflict: str = "retry",
        max_conflict_retries: int = 3,
    ) -> Callable:
        """
        Creates the actual task wrapper function.

        Args:
            func: The original function to wrap
            primer: Agent instructions for implementing the task (None to use docstring)
            on_conflict: How to handle concurrency conflicts ('retry' or 'abandon')
            max_conflict_retries: Maximum retry attempts for 'retry' strategy

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

        # Create new signature with added session parameter
        # Insert session parameter before **kwargs if it exists, otherwise append at end
        new_params = list(original_sig.parameters.values())
        session_param = inspect.Parameter(
            "session",
            inspect.Parameter.KEYWORD_ONLY,
            default="default",
            annotation="str",
        )
        on_event_param = inspect.Parameter(
            "on_event",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Callable[[BaseEvent], None] | None",
        )
        on_token_param = inspect.Parameter(
            "on_token",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation="Callable[[TokenChunk], None] | None",
        )

        # Find if there's a **kwargs parameter (VAR_KEYWORD)
        var_keyword_index = None
        for i, param in enumerate(new_params):
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                var_keyword_index = i
                break

        if var_keyword_index is not None:
            # Insert parameters before **kwargs
            new_params.insert(var_keyword_index, on_token_param)
            new_params.insert(var_keyword_index, on_event_param)
            new_params.insert(var_keyword_index, session_param)
        else:
            # No **kwargs, append at end
            new_params.append(session_param)
            new_params.append(on_event_param)
            new_params.append(on_token_param)

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
                # Expose only the original signature to agents (hide framework parameters)
                self.__annotations__ = func.__annotations__.copy()
                self.__signature__ = original_sig  # Use original, not new_sig

                # Store the full signature internally for argument binding
                self._internal_signature = new_sig

                # Set namespace for dual-decorator pattern
                namespace = self._agent_name
                self.__agex_task_namespace__ = namespace
                # Track if the underlying task is async for @remote decorator
                self.__agex_is_async__ = inspect.iscoroutinefunction(task_func)

            def __call__(self, *args, **kwargs):
                return self._task_func(*args, **kwargs)

            def __repr__(self):
                return f"<agex.task {self._agent_name}/{self._task_name} at {hex(id(self))}>"

            @property
            def stream(self):
                return self._stream_func

        # Helper to bind and validate arguments for both sync and async wrappers
        def _bind_and_validate(*args, **kwargs):
            # Pop internal _parent_state before binding (not part of public signature)
            parent_state = kwargs.pop("_parent_state", None)

            # Bind to the new signature that includes the 'session', 'on_event', and 'on_token' parameters
            bound_args = new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Pop the session, on_event, on_token arguments
            session = bound_args.arguments.pop("session", "default")
            on_event = bound_args.arguments.pop("on_event", None)
            on_token = bound_args.arguments.pop("on_token", None)

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

            return inputs_instance, session, on_event, on_token, parent_state

        # Create the actual task function (async or sync based on original function)
        if inspect.iscoroutinefunction(func):

            async def task_wrapper(*args, **kwargs):
                inputs_instance, session, on_event, on_token, parent_state = (
                    _bind_and_validate(*args, **kwargs)
                )
                # Route through host for non-local, top-level execution only
                # Sub-agent calls (parent_state is set) always execute locally
                if not isinstance(self._host, Local) and parent_state is None:
                    # Extract raw args/kwargs for remote execution
                    # (session, on_event, on_token already extracted by _bind_and_validate)
                    # Pop _parent_state before binding (it's not part of public signature)
                    binding_kwargs = {
                        k: v for k, v in kwargs.items() if k != "_parent_state"
                    }
                    bound = new_sig.bind(*args, **binding_kwargs)
                    bound.apply_defaults()
                    raw_kwargs = dict(bound.arguments)
                    raw_kwargs.pop("session", None)
                    raw_kwargs.pop("on_event", None)
                    raw_kwargs.pop("on_token", None)
                    return await self._host.aexecute(
                        agent=self,
                        task_name=task_name,
                        args=(),
                        kwargs=raw_kwargs,
                        session=session,
                        on_event=on_event,
                        on_token=on_token,
                    )
                # Use parent_state if provided (sub-task), otherwise resolve from session
                state = (
                    parent_state
                    if parent_state is not None
                    else self._host.resolve_state(self._state_config, session)
                )
                return await self._arun_task_loop(
                    task_name=task_name,
                    docstring=effective_docstring,
                    inputs_dataclass=inputs_dataclass,
                    inputs_instance=inputs_instance,
                    return_type=return_type,
                    state=state,
                    on_event=on_event,
                    on_token=on_token,
                    setup=setup,
                    on_conflict=on_conflict,
                    max_conflict_retries=max_conflict_retries,
                )

        else:

            def task_wrapper(*args, **kwargs):
                inputs_instance, session, on_event, on_token, parent_state = (
                    _bind_and_validate(*args, **kwargs)
                )
                # Route through host for non-local, top-level execution only
                # Sub-agent calls (parent_state is set) always execute locally
                if not isinstance(self._host, Local) and parent_state is None:
                    # Extract raw args/kwargs for remote execution
                    # (session, on_event, on_token already extracted by _bind_and_validate)
                    # Pop _parent_state before binding (it's not part of public signature)
                    binding_kwargs = {
                        k: v for k, v in kwargs.items() if k != "_parent_state"
                    }
                    bound = new_sig.bind(*args, **binding_kwargs)
                    bound.apply_defaults()
                    raw_kwargs = dict(bound.arguments)
                    raw_kwargs.pop("session", None)
                    raw_kwargs.pop("on_event", None)
                    raw_kwargs.pop("on_token", None)
                    return self._host.execute(
                        agent=self,
                        task_name=task_name,
                        args=(),
                        kwargs=raw_kwargs,
                        session=session,
                        on_event=on_event,
                        on_token=on_token,
                    )
                # Use parent_state if provided (sub-task), otherwise resolve from session
                state = (
                    parent_state
                    if parent_state is not None
                    else self._host.resolve_state(self._state_config, session)
                )
                return self._run_task_loop(
                    task_name=task_name,
                    docstring=effective_docstring,
                    inputs_dataclass=inputs_dataclass,
                    inputs_instance=inputs_instance,
                    return_type=return_type,
                    state=state,
                    on_event=on_event,
                    on_token=on_token,
                    setup=setup,
                    on_conflict=on_conflict,
                    max_conflict_retries=max_conflict_retries,
                )

        def stream(*args, **kwargs):
            """Stream events in real-time during task execution."""
            # Same parameter processing as regular task execution
            bound_args = new_sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Pop the session, on_event, and on_token arguments, they are handled separately
            session = bound_args.arguments.pop("session", "default")
            user_on_event = bound_args.arguments.pop("on_event", None)
            on_token = bound_args.arguments.pop("on_token", None)

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

            # Resolve state from host
            state = self._host.resolve_state(self._state_config, session)

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
                        on_token=on_token,
                        setup=setup,
                        on_conflict=on_conflict,
                        max_conflict_retries=max_conflict_retries,
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
        # Attach agent instance for @remote decorator access
        wrapper.__agex_agent__ = self  # type: ignore

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

        # Register for pickling (enables remote execution)
        _register_dataclass_for_pickling(inputs_dataclass, dataclass_name)

        # Register the dataclass with the agent for sandbox access
        if hasattr(self, "cls"):
            self.cls(inputs_dataclass, constructable=False)  # type: ignore

        return inputs_dataclass
