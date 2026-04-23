"""
Task decorator mixin for Agent class.

This module provides the TaskMixin that handles the @agent.task decorator
which wraps functions to become agent tasks.
"""

import inspect
import logging
from contextlib import contextmanager
from dataclasses import make_dataclass
from typing import Any, Callable

from kvgit import Staged

from agex.agent.base import BaseAgent
from agex.agent.datatypes import TaskClarify, TaskFail
from agex.agent.events import ActionEvent, OutputEvent
from agex.agent.loop import TaskLoopMixin
from agex.agent.utils import call_sync_or_async, is_function_body_empty
from agex.eval.objects import PrintAction
from agex.eval.validation import validate_with_sampling
from agex.host import Local
from agex.state import raw_set
from agex.state.log import add_event_to_log

logger = logging.getLogger(__name__)

# Global registry for dynamically created input dataclasses
# This allows pickle to find them by module.classname lookup
_DYNAMIC_DATACLASS_REGISTRY: dict[str, type] = {}


@contextmanager
def _report_tap(on_event, agent_name):
    """Context manager that wraps an ``on_event`` callback to collect
    sub-agent REPORTs during a task call, then flushes them into the
    parent's state log on exit.

    Yields a replacement ``on_event`` callback that forwards every event
    to the original while also collecting ``ActionEvent.report`` values
    attributed to *agent_name*.

    On exit (normal return or exception), any collected reports are
    injected as a synthetic ``OutputEvent`` into the parent's log via
    ``_current_parent_log``.  At the top level (no parent sandbox
    active), the flush is a no-op — subscribers still see the reports
    via the sub-agent's own ``ActionEvent.report`` flowing through the
    shared ``on_event`` callback.

    The tap is always a sync function.  For async user callbacks it
    returns the unawaited coroutine; call sites that use
    ``call_sync_or_async`` + ``await`` handle this correctly, and
    direct-sync call sites drop it (pre-existing behavior).
    """
    collected: list[str] = []

    def tapped(event):
        # Agent-name filter is critical for multi-level nesting:
        # without it, a grandparent's tap would also collect a
        # grandchild's reports via the callback chain and
        # double-emit into the wrong parent's log.
        if isinstance(event, ActionEvent) and event.agent_name == agent_name:
            from agex.agent.emissions import TextEmission

            texts = [
                em.text
                for em in event.emissions
                if isinstance(em, TextEmission) and em.text
            ]
            if texts:
                collected.append("\n\n".join(texts))
        if on_event is not None:
            return call_sync_or_async(on_event, event)
        return None

    try:
        yield tapped
    finally:
        if collected:
            from agex.eval.bridge.policy import _current_parent_log

            parent_log = _current_parent_log.get()
            if parent_log is not None:
                parent_state, parent_name = parent_log
                try:
                    output_event = OutputEvent(
                        agent_name=parent_name,
                        parts=[
                            PrintAction([f"[report:{agent_name}] {text}"])
                            for text in collected
                        ],
                    )
                    # on_event=None: subscribers already saw these reports
                    # via the sub-agent's ActionEvent flowing through the
                    # tap chain.  Firing a duplicate synthetic event would
                    # cause double notification.
                    add_event_to_log(parent_state, output_event, on_event=None)
                except Exception:
                    # Never mask the real return value / exception from
                    # the task loop with a log-write failure.
                    logger.debug(
                        "Failed to inject sub-agent REPORTs into parent log",
                        exc_info=True,
                    )


def _reactivate_result(result, agent):
    """Reactivate sandbox wrappers returned from external execution.

    When StFunction/StClass/StInstance objects cross a process boundary
    (process isolation, Modal, HTTP), they arrive inactive.  This rebuilds
    gates from the agent's policy and reactivates them so they're callable.
    """
    from sandtrap.wrappers import StClass, StFunction, StInstance, activate_value

    if isinstance(result, (StFunction, StClass, StInstance)):
        needs_activation = True
    elif isinstance(result, (list, tuple)):
        needs_activation = any(
            isinstance(item, (StFunction, StClass, StInstance)) for item in result
        )
    elif isinstance(result, dict):
        needs_activation = any(
            isinstance(v, (StFunction, StClass, StInstance)) for v in result.values()
        )
    else:
        return result

    if not needs_activation:
        return result

    from sandtrap.gates import make_gates

    from agex.eval.bridge.policy import translate_policy

    policy = translate_policy(agent, timeout=None, tick_limit=None)
    gates = make_gates(policy)

    if isinstance(result, (StFunction, StClass, StInstance)):
        activate_value(result, gates)
    elif isinstance(result, dict):
        for v in result.values():
            if isinstance(v, (StFunction, StClass, StInstance)):
                activate_value(v, gates)
    else:
        for item in result:
            if isinstance(item, (StFunction, StClass, StInstance)):
                activate_value(item, gates)

    return result


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
        session: str,
        on_event: Callable[[Any], None] | None = None,
        on_token: Callable[[Any], None] | None = None,
    ) -> Any:
        """
        Execute a sub-agent task with inherited session.

        Each sub-agent resolves its own state using its state config and the
        inherited session. This ensures state isolation while maintaining
        session continuity across the agent hierarchy.

        Args:
            task_callable: The callable produced by @agent.task
            args: Positional arguments to pass to the task
            kwargs: Keyword arguments to pass to the task
            session: Session identifier inherited from parent
            on_event: Optional event handler to propagate
            on_token: Optional token handler to propagate

        Returns:
            The task result produced by the task loop
        """
        # Rehydrate sub-agent if needed (may have llm=None, _host=None after deserialization)
        # This happens when cloudpickle deserializes nested agents in closures
        sub_agent = getattr(task_callable, "__agex_agent__", None)
        if sub_agent is not None:
            # Rehydrate LLM from serialized config if missing
            if sub_agent.llm is None:
                if hasattr(sub_agent, "_llm_config") and sub_agent._llm_config:
                    from agex.llm import LLM

                    sub_agent.llm = LLM.from_config(sub_agent._llm_config)

            # Rehydrate Host from serialized config if missing
            if sub_agent._host is None:
                if hasattr(sub_agent, "_host_config") and sub_agent._host_config:
                    from agex.host.base import Host

                    sub_agent._host = Host.from_config(sub_agent._host_config)
                else:
                    # Default to Local if no config
                    from agex.host import Local

                    sub_agent._host = Local()

            # Fingerprint recomputes lazily on first access

        # Prepare kwargs - pass session for state resolution
        call_kwargs = dict(kwargs) if kwargs is not None else {}
        call_kwargs["session"] = session
        if on_event is not None:
            call_kwargs["on_event"] = on_event
        if on_token is not None:
            call_kwargs["on_token"] = on_token

        # Execute sub-agent task, converting TaskClarify/TaskFail to EvalError
        # Sub-agents can't request user clarification - parent sees error instead
        try:
            return task_callable(*args, **call_kwargs)
        except TaskClarify as e:
            from agex.eval.error import EvalError

            raise EvalError(f"Sub-agent needs clarification: {e.message}") from e
        except TaskFail as e:
            from agex.eval.error import EvalError

            raise EvalError(f"Sub-agent failed: {e.message}") from e

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
            def __init__(self, task_func, agent_name, task_name):
                self._task_func = task_func
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
                # Allow network access for task calls (needed for LLM calls in sub-agent tasks)
                self.network_access = True

            def __call__(self, *args, **kwargs):
                return self._task_func(*args, **kwargs)

            def __repr__(self):
                return f"<agex.task {self._agent_name}/{self._task_name} at {hex(id(self))}>"

            def cancel(self, session: str = "default") -> None:
                """
                Request cancellation of this task on the given session.

                Writes a cancellation sentinel to state that the task loop
                will check between iterations. The task will stop gracefully
                at the next checkpoint.

                Args:
                    session: Session identifier (default: "default")

                Raises:
                    NotImplementedError: If the host doesn't support client-side state access
                    ValueError: If the state doesn't exist yet
                """
                agent = getattr(self, "__agex_agent__", None)
                if agent is None:
                    raise RuntimeError("TaskWrapper has no associated agent")

                state = agent.state(session)
                if state is None:
                    raise RuntimeError(
                        f"Cannot cancel task '{self._task_name}': agent has no state configured. "
                        "Cancellation requires state=connect_state(...) on the agent."
                    )

                # Write task-specific cancellation sentinel
                cancel_key = f"__agex_cancel__{self._task_name}"

                # Write directly to underlying KV store for immediate visibility
                # (bypasses versioned commits so running tasks can see it)
                if isinstance(state, Staged):
                    raw_set(state, cancel_key, True)
                else:
                    # Live state - just set normally
                    state[cancel_key] = True

        # Helper to bind and validate arguments for both sync and async wrappers
        def _bind_and_validate(*args, **kwargs):
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

            return inputs_instance, session, on_event, on_token

        # Sync task function — used both as the primary callable for sync tasks
        # and as _sync_task_func for async tasks (so sub-agent calls from
        # sandbox code can avoid event-loop conflicts).
        def sync_task_func(*args, **kwargs):
            inputs_instance, session, on_event, on_token = _bind_and_validate(
                *args, **kwargs
            )
            with _report_tap(on_event, self.name) as tapped_on_event:
                # Route through host for non-local execution
                if not isinstance(self._host, Local):
                    # Extract raw args/kwargs for remote execution
                    bound = new_sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    raw_kwargs = dict(bound.arguments)
                    raw_kwargs.pop("session", None)
                    raw_kwargs.pop("on_event", None)
                    raw_kwargs.pop("on_token", None)
                    return _reactivate_result(
                        self._host.execute(
                            agent=self,
                            task_name=task_name,
                            args=(),
                            kwargs=raw_kwargs,
                            session=session,
                            on_event=tapped_on_event,
                            on_token=on_token,
                        ),
                        self,
                    )
                # Resolve state from session using agent's state config
                state = self._host.resolve_state(
                    self._state_config, session, self.fingerprint or ""
                )
                return self._run_task_loop(
                    task_name=task_name,
                    docstring=effective_docstring,
                    inputs_dataclass=inputs_dataclass,
                    inputs_instance=inputs_instance,
                    return_type=return_type,
                    state=state,
                    session=session,
                    on_event=tapped_on_event,
                    on_token=on_token,
                    setup=setup,
                    on_conflict=on_conflict,
                    max_conflict_retries=max_conflict_retries,
                )

        # Create the actual task function (async or sync based on original function)
        if inspect.iscoroutinefunction(func):

            async def task_wrapper(*args, **kwargs):
                inputs_instance, session, on_event, on_token = _bind_and_validate(
                    *args, **kwargs
                )
                with _report_tap(on_event, self.name) as tapped_on_event:
                    # Route through host for non-local execution
                    if not isinstance(self._host, Local):
                        # Extract raw args/kwargs for remote execution
                        bound = new_sig.bind(*args, **kwargs)
                        bound.apply_defaults()
                        raw_kwargs = dict(bound.arguments)
                        raw_kwargs.pop("session", None)
                        raw_kwargs.pop("on_event", None)
                        raw_kwargs.pop("on_token", None)
                        return _reactivate_result(
                            await self._host.aexecute(
                                agent=self,
                                task_name=task_name,
                                args=(),
                                kwargs=raw_kwargs,
                                session=session,
                                on_event=tapped_on_event,
                                on_token=on_token,
                            ),
                            self,
                        )
                    # Resolve state from session using agent's state config
                    state = self._host.resolve_state(
                        self._state_config, session, self.fingerprint or ""
                    )
                    return await self._arun_task_loop(
                        task_name=task_name,
                        docstring=effective_docstring,
                        inputs_dataclass=inputs_dataclass,
                        inputs_instance=inputs_instance,
                        return_type=return_type,
                        state=state,
                        session=session,
                        on_event=tapped_on_event,
                        on_token=on_token,
                        setup=setup,
                        on_conflict=on_conflict,
                        max_conflict_retries=max_conflict_retries,
                    )

        else:
            task_wrapper = sync_task_func

        # Create the custom wrapper with proper __repr__
        agent_name = self.name if self.name is not None else self.__class__.__name__
        wrapper = TaskWrapper(task_wrapper, agent_name, task_name)
        # Store sync version for sub-agent sandbox calls (avoids event-loop conflicts)
        wrapper._sync_task_func = sync_task_func
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
