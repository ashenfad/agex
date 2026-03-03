# Main agent functionality
from typing import TYPE_CHECKING

from ..host import Host
from ..llm import LLM
from .base import (
    _UNSET,
    BaseAgent,
    Isolation,
    clear_agent_registry,
)
from .chapter import CHAPTER_TASK, CHAPTER_TASK_PRIMER, Chapter

if TYPE_CHECKING:
    from agex.fs import FSConfig

    from ..state.config import StateConfig

# Data types and exceptions
from .datatypes import (
    RESERVED_NAMES,
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
from .task import TaskMixin, clear_dynamic_dataclass_registry

__all__ = [
    # Core functionality
    "clear_agent_registry",
    "clear_dynamic_dataclass_registry",
    # Task control functions
    "TaskSuccess",
    "TaskFail",
    "TaskContinue",
    # Registration types
    "MemberSpec",
    "RegisteredItem",
    "RegisteredFn",
    "RegisteredClass",
    "RegisteredModule",
    # Type aliases and constants
    "Pattern",
    "Visibility",
    "RESERVED_NAMES",
    # Chapter constants
    "CHAPTER_TASK",
    # Exceptions
    # Fingerprinting
]


class Agent(RegistrationMixin, TaskMixin, TaskLoopMixin, BaseAgent):
    @classmethod
    def clone_registrations(
        cls,
        source: "Agent",
        *,
        name: str | None = None,
        primer: str | None = None,
        eval_timeout_seconds: float = 5.0,
        max_iterations: int = 10,
        llm: LLM | None = None,
        host: Host | None = None,
        state: "StateConfig | None" = None,
        fs: "FSConfig | None" = _UNSET,  # type: ignore
        max_memory_mb: int | None = None,
        max_open_files: int | None = None,
        eval_tick_limit: int | None = 100_000,
        isolation: "Isolation" = "none",
    ) -> "Agent":
        """
        Create a new agent with copied registrations but independent state/fs/host.

        The new agent inherits all registrations (modules, functions, classes)
        from the source agent but has its own state, filesystem, and host.
        Additional registrations can be added without affecting the source.

        This is useful for creating "sandbox" agents that need the same capabilities
        as a main agent (e.g., for executing user-generated code with access to
        registered libraries) but with isolated state.

        Args:
            source: The agent to copy registrations from
            name: Optional name for the new agent
            primer: Optional primer string (defaults to None)
            eval_timeout_seconds: Code execution timeout (defaults to 5.0)
            max_iterations: Max think-act cycles (defaults to 10)
            llm: LLM configuration (defaults to None)
            host: Execution host (defaults to Local())
            state: State configuration (defaults to ephemeral)
            fs: FileSystem configuration (defaults to VirtualFS)

        Returns:
            A new Agent with copied registrations and independent state/fs/host

        Example:
            # Main agent with domain capabilities
            main_agent = Agent(...)
            main_agent.module(pandas)
            main_agent.module(plotly)

            # Sandbox with same capabilities plus UI, isolated state
            sandbox = Agent.clone_registrations(
                main_agent,
                name="sandbox",
                state=connect_state(type="versioned", storage="memory"),
            )
            sandbox.module(ui)  # Doesn't affect main_agent
        """
        new_agent = cls(
            name=name,
            primer=primer,
            eval_timeout_seconds=eval_timeout_seconds,
            max_iterations=max_iterations,
            llm=llm,
            host=host,
            state=state,
            fs=fs,
            max_memory_mb=max_memory_mb,
            max_open_files=max_open_files,
            eval_tick_limit=eval_tick_limit,
            isolation=isolation,
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
        # Event log chaptering (optional)
        log_high_water_tokens: int | None = None,
        log_low_water_tokens: int | None = None,
        # Resource limits (per-task, Unix only)
        max_memory_mb: int | None = None,
        max_open_files: int | None = None,
        # Tick-based execution limit (primary runaway-code protection)
        eval_tick_limit: int | None = 100_000,
        # Sandbox isolation level (passed to sandtrap)
        isolation: "Isolation" = "none",
        # Advanced: Override the built-in system instructions
        agex_primer_override: str | None = None,
    ):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            eval_timeout_seconds: The maximum time in seconds for agent-generated code to run.
                When eval_tick_limit is set, this is replaced by a generous 300s safety net.
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
            log_high_water_tokens: Trigger chaptering when input tokens exceed this
                threshold. If None, chaptering is disabled.
            log_low_water_tokens: Stop chaptering once input tokens drop below this
                threshold. Defaults to 50% of log_high_water_tokens if not specified.
            max_memory_mb: Per-task memory limit in megabytes. Passed to sandtrap's
                Policy.memory_limit. Kernel-enforced on Linux, checkpoint-based on macOS.
            max_open_files: Maximum file descriptors for the process. Unix only.
            eval_tick_limit: Maximum number of Python control-flow checkpoints
                (loop iterations, function entries, comprehensions) per code execution.
                Defaults to 100,000. Set to None to disable and rely solely on
                eval_timeout_seconds.
            isolation: Sandbox isolation level. "none" (default) runs in-process.
                "process" runs in a subprocess for crash protection. "kernel" adds
                kernel-level filesystem, syscall, and network restrictions.
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
            max_memory_mb=max_memory_mb,
            max_open_files=max_open_files,
            eval_tick_limit=eval_tick_limit,
            isolation=isolation,
            agex_primer_override=agex_primer_override,
        )

        # Register chapter support if water marks are configured
        self._chapter_task = None
        if self.log_high_water_tokens is not None:
            self._register_chapter_task()

    def _register_chapter_task(self):
        """Register the Chapter class and chapter task for context compaction."""
        # Register Chapter class so agent can construct instances
        self.cls(Chapter, constructable=True)

        # Register the chapter task
        def __chapter__(event_index: str) -> list:
            pass

        __chapter__.__doc__ = CHAPTER_TASK_PRIMER
        self._chapter_task = self.task(__chapter__)
