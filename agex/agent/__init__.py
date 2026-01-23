# Main agent functionality
from typing import TYPE_CHECKING

from ..host import Host
from ..llm import LLM
from .base import (
    _UNSET,
    BaseAgent,
    clear_agent_registry,
    register_agent,
    resolve_agent,
)

if TYPE_CHECKING:
    from ..fs.config import FSConfig
    from ..state.config import StateConfig

# Data types and exceptions
from .datatypes import (
    RESERVED_NAMES,
    AttrDescriptor,
    MemberSpec,
    Pattern,
    RegisteredClass,
    RegisteredFn,
    RegisteredItem,
    RegisteredModule,
    TaskContinue,
    TaskFail,
    TaskSuccess,
    Visibility,
)
from .loop import TaskLoopMixin

# Fingerprinting (usually internal, but exported for testing)
from .registration import RegistrationMixin
from .summarization import SummarizationError
from .task import TaskMixin, clear_dynamic_dataclass_registry

__all__ = [
    # Core functionality
    "register_agent",
    "resolve_agent",
    "clear_agent_registry",
    "clear_dynamic_dataclass_registry",
    # Task control functions
    "TaskSuccess",
    "TaskFail",
    "TaskContinue",
    # Registration types
    "MemberSpec",
    "AttrDescriptor",
    "RegisteredItem",
    "RegisteredFn",
    "RegisteredClass",
    "RegisteredModule",
    # Type aliases and constants
    "Pattern",
    "Visibility",
    "RESERVED_NAMES",
    # Exceptions
    "SummarizationError",
    # Fingerprinting
]


class Agent(RegistrationMixin, TaskMixin, TaskLoopMixin, BaseAgent):
    @classmethod
    def clone(cls, source: "Agent", *, name: str | None = None) -> "Agent":
        """
        Create a new agent with copied policy but shared state/fs/host.

        The cloned agent inherits all registrations (modules, functions, classes)
        from the source agent but can have additional registrations added without
        affecting the source.

        State, filesystem, and host are shared, so both agents access the same
        underlying state and files for a given session. This enables use cases
        like running user-generated code in a sandbox with access to files
        written by the main agent.

        This is useful for creating "sandbox" agents that need the same capabilities
        as a main agent plus additional modules (e.g., for executing user-generated
        code with access to registered libraries plus UI frameworks).

        Args:
            source: The agent to clone from
            name: Optional name for the new agent (defaults to "{source.name}_clone")

        Returns:
            A new Agent with copied policy and shared state/fs/host

        Example:
            # Main agent with domain capabilities
            main_agent = Agent(...)
            main_agent.module(calgebra)
            main_agent.module(pandas)

            # Sandbox with same capabilities plus UI
            sandbox = Agent.clone(main_agent, name="sandbox")
            sandbox.module(ui)  # Doesn't affect main_agent

            # Files written by main_agent are visible to sandbox
            main_agent.fs("session").write("app/main.py", code)
            run_file_in_sandbox(sandbox, "app/main.py", "session")
        """
        # Create new agent with shared state/fs
        # We pass the source's host to share the session cache (needed for shared VFS)
        new_agent = cls(
            name=name or f"{source.name}_clone",
            primer=source.primer,
            eval_timeout_seconds=source.eval_timeout_seconds,
            max_iterations=source.max_iterations,
            # Share state and fs configurations
            state=source._state_config,
            fs=source._fs_config,
            # Share host to ensure same session cache (important for VFS sharing)
            host=source._host,
        )

        # Copy the policy so modifications don't affect source
        # The copy shares references to live objects (modules, functions, classes)
        # but has independent dict structures for adding new registrations
        new_agent._policy = source._policy.copy()

        # Copy the host object registry (live instances)
        new_agent._host_object_registry = source._host_object_registry.copy()

        # Copy tracked modules for dependency resolution
        new_agent._tracked_modules = source._tracked_modules.copy()

        # Update fingerprint after copying
        new_agent._update_fingerprint()

        return new_agent

    def __init__(
        self,
        primer: str | None = None,
        eval_timeout_seconds: float = 5.0,
        max_iterations: int = 10,
        # Agent identification
        name: str | None = None,
        # Optional curated capabilities primer
        capabilities_primer: str | None = None,
        # LLM configuration (optional, uses smart defaults)
        llm: LLM | None = None,
        # LLM retry control (timeout comes from llm.timeout_seconds)
        llm_max_retries: int = 2,
        # Host configuration (optional, defaults to local execution)
        host: Host | None = None,
        # State configuration (optional, defaults to ephemeral)
        state: "StateConfig | None" = None,
        # FileSystem configuration (optional, defaults to VirtualFS)
        fs: "FSConfig | None" = _UNSET,  # type: ignore
        # Event log summarization (optional)
        log_high_water_tokens: int | None = None,
        log_low_water_tokens: int | None = None,
        # Advanced: Override the built-in system instructions
        agex_primer_override: str | None = None,
    ):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            eval_timeout_seconds: The maximum time in seconds for agent-generated code to run.
            max_iterations: The maximum number of think-act cycles for a task.
            name: Unique identifier for this agent (for sub-agent namespacing).
            capabilities_primer: Optional curated capabilities primer.
            llm: An instantiated LLM for the agent to use. Configure
                llm.timeout_seconds for API timeout control.
            llm_max_retries: Number of retry attempts for failed/timed-out LLM calls.
            host: Execution host for tasks. Defaults to Local() which runs in-process.
                Use HTTP(url=...) for remote execution.
            state: State configuration (optional). Use connect_state() to create.
                Defaults to ephemeral (fresh state per task call).
            fs: FileSystem configuration (optional). Use connect_fs() to create.
                Enables virtual filesystem access for agents.
            log_high_water_tokens: Trigger event log summarization when total tokens
                exceed this threshold. If None, no summarization is performed.
            log_low_water_tokens: Target token count after summarization. Defaults to
                50% of log_high_water_tokens if not specified.
            agex_primer_override: (Advanced) Override the built-in system instructions
                that define the agent's core behavior and event protocol.
        """
        super().__init__(
            primer=primer,
            eval_timeout_seconds=eval_timeout_seconds,
            max_iterations=max_iterations,
            name=name,
            capabilities_primer=capabilities_primer,
            llm=llm,
            llm_max_retries=llm_max_retries,
            host=host,
            state=state,
            fs=fs,
            log_high_water_tokens=log_high_water_tokens,
            log_low_water_tokens=log_low_water_tokens,
            agex_primer_override=agex_primer_override,
        )
        # Track external package dependencies incrementally
        self._dependencies: set[str] = set()
