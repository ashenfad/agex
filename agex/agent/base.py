import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, Literal

from ..llm import LLM, connect_llm
from ..resource_limits import ResourceLimits
from .datatypes import MemberSpec, RegisteredClass
from .fingerprint import compute_agent_fingerprint_from_policy
from .policy.policy import AgentPolicy

if TYPE_CHECKING:
    from ..fs.aware import AgentAwareFS
    from ..fs.config import FSConfig
    from ..host import Host
    from ..host.dependencies import Dependencies
    from ..state import State
    from ..state.config import StateConfig

# Global registry mapping fingerprints to agents
# Using process-global dicts with lock for thread safety
# (ContextVar doesn't work for callbacks invoked in different async contexts)
_AGENT_REGISTRY: Dict[str, "BaseAgent"] = {}
_AGENT_REGISTRY_BY_NAME: Dict[str, "BaseAgent"] = {}
_AGENT_REGISTRY_LOCK = threading.Lock()

_UNSET = object()


def register_agent(agent: "BaseAgent") -> str:
    """
    Register an agent in the global registry.

    Returns the agent's fingerprint.

    Note: Allows re-registration of agents with the same fingerprint,
    which is necessary for handling deserialized agents (e.g., in remote
    execution where the same agent may be sent multiple times).
    """
    # Compute fingerprint first - needed for collision detection
    fingerprint = compute_agent_fingerprint_from_policy(agent)

    with _AGENT_REGISTRY_LOCK:
        # Enforce unique agent names if provided
        if hasattr(agent, "name") and agent.name is not None:
            if agent.name in _AGENT_REGISTRY_BY_NAME:
                existing_agent = _AGENT_REGISTRY_BY_NAME[agent.name]
                # Allow re-registration if:
                # 1. Same object instance (identity), OR
                # 2. Same fingerprint (deserialized copy of same agent)
                # 3. Existing agent has no fingerprint (shouldn't happen but defensive)
                existing_fingerprint = getattr(existing_agent, "fingerprint", None)

                if existing_agent is not agent:
                    if (
                        existing_fingerprint is None
                        or existing_fingerprint != fingerprint
                    ):
                        raise ValueError(f"Agent name '{agent.name}' already exists")
            _AGENT_REGISTRY_BY_NAME[agent.name] = agent

        _AGENT_REGISTRY[fingerprint] = agent

    return fingerprint


def resolve_agent(fingerprint: str) -> "BaseAgent":
    """
    Resolve an agent by its fingerprint.

    Raises RuntimeError if no matching agent is found.
    """
    with _AGENT_REGISTRY_LOCK:
        agent = _AGENT_REGISTRY.get(fingerprint)
        if not agent:
            available = list(_AGENT_REGISTRY.keys())
            raise RuntimeError(
                f"No agent found with fingerprint '{fingerprint[:8]}...'. "
                f"Available fingerprints: {[fp[:8] + '...' for fp in available]}"
            )
        return agent


def clear_agent_registry() -> None:
    """Clear the global registry. Primarily for testing."""
    from .task import clear_dynamic_dataclass_registry

    with _AGENT_REGISTRY_LOCK:
        _AGENT_REGISTRY.clear()
        _AGENT_REGISTRY_BY_NAME.clear()
    clear_dynamic_dataclass_registry()


def get_agent_by_name(name: str) -> "BaseAgent | None":
    """Get an agent by its name.

    Used for fallback resolution when an agent's fingerprint has changed
    (e.g., after config/primer changes) but the agent name is still the same.

    Returns:
        The agent with the given name, or None if not found.
    """
    with _AGENT_REGISTRY_LOCK:
        return _AGENT_REGISTRY_BY_NAME.get(name)


def _random_name() -> str:
    return f"agent_{uuid.uuid4().hex[:8]}"


class BaseAgent:
    def __init__(
        self,
        primer: str | None,
        eval_timeout_seconds: float,
        max_iterations: int,
        # Agent identification
        name: str | None = None,
        # Optional curated capabilities primer (overrides rendered registrations when set)
        capabilities_primer: str | None = None,
        # LLM configuration (optional, uses smart defaults)
        llm: LLM | None = None,
        # LLM retry control (timeout comes from llm.timeout_seconds)
        llm_max_retries: int = 2,
        # Host configuration (optional, defaults to local execution)
        host: "Host | None" = None,
        # State configuration (optional, defaults to ephemeral)
        state: "StateConfig | None" = None,
        # FileSystem configuration (optional, defaults to VirtualFS)
        fs: "FSConfig | None" = _UNSET,  # type: ignore
        # Event log summarization (optional)
        log_high_water_tokens: int | None = None,
        log_low_water_tokens: int | None = None,
        # Resource limits (per-task, Unix only)
        max_memory_mb: int | None = None,
        max_open_files: int | None = None,
        # Advanced: Override the builtin system instructions
        agex_primer_override: str | None = None,
    ):
        self.name = name or _random_name()
        self.primer = primer
        # If set, used instead of rendered registrations in system context
        self.capabilities_primer = capabilities_primer
        # Advanced: Override the builtin system instructions
        self.agex_primer_override = agex_primer_override
        self.eval_timeout_seconds = eval_timeout_seconds
        self.max_iterations = max_iterations

        # Create LLM using the resolved configuration
        self.llm = llm or connect_llm()
        # Store LLM config for serialization and dependency inference
        self._llm_config = (
            self.llm.dump_config()
            if self.llm and hasattr(self.llm, "dump_config")
            else None
        )
        # LLM retry setting (timeout comes from llm.timeout_seconds)
        self.llm_max_retries = llm_max_retries

        # Execution host (defaults to local)
        from ..host import Local

        self._host: "Host" = host or Local()

        # State configuration (None = ephemeral)
        self._state_config: "StateConfig | None" = state

        # FileSystem configuration (defaults to VirtualFS)
        from ..fs import connect_fs

        if fs is _UNSET:
            self._fs_config: "FSConfig | None" = connect_fs(type="virtual")
        else:
            self._fs_config = fs

        # Validate state config is compatible with the host
        if self._state_config is not None:
            self._host.validate_state(self._state_config)

        # Event log summarization settings
        if log_low_water_tokens is not None and log_high_water_tokens is None:
            raise ValueError(
                "log_low_water_tokens requires log_high_water_tokens to be set"
            )

        if log_high_water_tokens is not None:
            if log_low_water_tokens is None:
                log_low_water_tokens = int(log_high_water_tokens * 0.5)
            elif log_low_water_tokens >= log_high_water_tokens:
                raise ValueError(
                    f"log_low_water_tokens ({log_low_water_tokens}) must be < "
                    f"log_high_water_tokens ({log_high_water_tokens})"
                )

        self.log_high_water_tokens = log_high_water_tokens
        self.log_low_water_tokens = log_low_water_tokens

        # Resource limits (per-task, Unix only)
        self._resource_limits = ResourceLimits(
            max_memory_mb=max_memory_mb,
            max_open_files=max_open_files,
        )

        # private, host-side registry for live, unpickleable objects
        self._host_object_registry: dict[str, Any] = {}

        self._policy: AgentPolicy = AgentPolicy()

        # Registry for tasks defined via @agent.task
        self._tasks: dict[str, Callable] = {}

        # Dependency tracking (lazy evaluation)
        self._tracked_modules: set[str] = (
            set()
        )  # Module names collected at registration
        self._cached_dependencies: "Dependencies | None" = None  # Computed on access

        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self._cached_dependencies = None  # Invalidate dependency cache
        self.fingerprint = register_agent(self)

    def __getstate__(self) -> dict[str, Any]:
        """
        Custom pickling state to handle runtime objects.

        Excludes:
        - _host_object_registry: Holds live instances (db connections, etc.)
        - llm: Live LLM has nonserializable state (sockets, SSL context)
        - _host: Host has non-serializable state (HTTP clients, etc.)
        - fingerprint: Computed from runtime state, might differ on host

        Adds:
        - _llm_config: Reconstructable configuration for the LLM
        - _host_config: Reconstructable configuration for the Host
        """
        state = self.__dict__.copy()

        # Serialize LLM config
        if self.llm:
            state["_llm_config"] = self.llm.dump_config()

        # Serialize Host config
        if self._host:
            state["_host_config"] = self._host.dump_config()

        # Remove runtime-only objects
        state.pop("llm", None)
        state.pop("_host", None)
        state.pop("_host_object_registry", None)
        state.pop("fingerprint", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore agent from pickle state.

        NOTE: The agent is not fully functional until rehydrated by prepare_agent,
        which must:
        1. Reconstruct the LLM from _llm_config
        2. Reconstruct the Host from _host_config
        3. Recompute fingerprint/register
        """
        # Restore configuration
        self.__dict__.update(state)

        # Initialize runtime fields
        self._host_object_registry = {}  # Empty on new host

        # llm and _host remain None until rehydrated by prepare_agent
        self.llm = None
        self._host = None
        self.fingerprint = None  # Will be recomputed

    def module(
        self,
        obj: Any,
        *,
        name: str | None = None,
        visibility: Literal["high", "medium", "low"] = "medium",
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        configure: dict[str, MemberSpec | RegisteredClass] | None = None,
    ) -> None:
        """
        Stub implementation of module registration.
        The full implementation with include/exclude support is in RegistrationMixin.
        This method should not be called directly - use Agent class instead.
        """
        raise NotImplementedError(
            "This is a stub implementation. Use the Agent class which inherits from "
            "RegistrationMixin for full include/exclude support."
        )

    def task(self, prompt: str | Callable) -> Callable[..., Any]: ...

    def warmup(self) -> None:
        """
        Pre-warm the execution host for faster cold starts.

        For serverless hosts (Modal, Beam), this builds the container image
        with inferred dependencies and starts a warm instance. For local/HTTP
        hosts, this is a no-op.

        Example:
            agent = Agent(
                host=connect_host(provider="modal", app="my-app"),
                ...
            )
            agent.warmup()  # Build image, start warm container
            result = my_task()  # Fast execution, no cold start
        """
        # Import here to avoid circular dependency at module level
        from agex.agent.registration import RegistrationMixin

        # Get dependencies from the agent (requires RegistrationMixin)
        if isinstance(self, RegistrationMixin):
            deps = self.dependencies
        else:
            # Fallback for pure BaseAgent (unlikely in practice)
            from importlib import metadata

            from agex.host.dependencies import Dependencies

            deps = Dependencies(
                python_version=f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
                agex_version=metadata.version("agex"),
                packages=[],
            )

        self._host.warmup(deps)

    def state(self, session: str = "default") -> "State":
        """
        Get the state object for a session.

        This allows client-side access to the same state used by task execution,
        enabling operations like:
        - Inspecting state with view() and events()
        - Rolling back to previous commits
        - Cancelling running tasks

        Example:
            from agex import Agent, connect_state, view, events

            agent = Agent(state=connect_state(type="versioned", storage="memory"))

            @agent.task
            def chat(message: str) -> str:
                pass

            # Execute some tasks
            chat("Hello")
            chat("How are you?")

            # Inspect state
            state = agent.state()
            print(view(state))
            print(events(state))

        Args:
            session: Session identifier (default: "default")

        Returns:
            The state object for this session

        Raises:
            NotImplementedError: If the host doesn't support client-side state access
            ValueError: If the state doesn't exist yet (run a task first)
        """
        return self._host.state(self._state_config, session, self.fingerprint or "")

    def fs(self, session: str = "default") -> "AgentAwareFS":
        """
        Get filesystem accessor for a session.

        This allows external code (like UI) to read/write files to the
        filesystem that is accessible to agent tasks.

        For virtual fs: Files are stored in agent state and participate in versioning.
        For isolated fs: Files are in the real filesystem, restricted to root directory.

        File operations via this accessor automatically emit FileEvents
        so agents can see external file changes in their context.

        Example:
            from agex import Agent, connect_fs, connect_state

            # Virtual filesystem
            agent = Agent(
                state=connect_state(type="versioned", storage="disk", path="/tmp/state"),
                fs=connect_fs(type="virtual"),
            )

            # Isolated filesystem
            agent = Agent(
                fs=connect_fs(type="isolated", root="/project", tracking=True),
            )

            # Upload a file from UI
            fs = agent.fs()
            fs.write("shared/data.csv", csv_bytes)

            # File is now accessible to agent tasks via open()

        Args:
            session: Session identifier (default: "default")

        Returns:
            AgentAwareFS interface with read(), write(), list(), exists(), remove() methods
        """
        from agex.fs import AgentAwareFS

        backend, state = self._get_fs_backend(session)
        if backend is None or state is None:
            raise ValueError("Filesystem is disabled for this agent")

        return AgentAwareFS(backend, state, self.name)

    def _get_fs_backend(self, session: str) -> tuple[Any, Any]:
        """Resolve the underlying filesystem backend and state for a session."""
        if not self._fs_config:
            return None, None

        from agex.fs import IsolatedFS, VirtualFS
        from agex.fs.config import IsolatedFSConfig, VirtualFSConfig

        state = self.state(session)

        if isinstance(self._fs_config, VirtualFSConfig):
            return VirtualFS(state, max_size_mb=self._fs_config.max_size_mb), state

        elif isinstance(self._fs_config, IsolatedFSConfig):
            from agex.eval.core import _get_session_root

            # Get session-specific root if per_session is enabled
            root = _get_session_root(
                self._fs_config.root, session, self._fs_config.per_session
            )

            return IsolatedFS(root, state), state

        else:
            raise ValueError(f"Unsupported filesystem config: {type(self._fs_config)}")

    def _fs_exists(self, filename: str, session: str) -> bool:
        """Check if a file exists in the session's filesystem without wrapping."""
        backend, _ = self._get_fs_backend(session)
        return backend.exists(filename) if backend else False

    def _fs_read(self, filename: str, session: str) -> bytes:
        """Read a file from the session's filesystem without wrapping."""
        backend, _ = self._get_fs_backend(session)
        if not backend:
            raise ValueError("Filesystem is disabled for this agent")
        return backend.read(filename)
