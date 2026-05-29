import os
import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

from ..llm import LLM, connect_llm
from ..resource_limits import ResourceLimits
from .datatypes import MemberSpec, RegisteredClass
from .fingerprint import compute_agent_fingerprint_from_policy
from .policy.policy import AgentPolicy

# Valid isolation levels for sandtrap's sandbox() factory
Isolation = Literal["none", "process", "kernel"]

if TYPE_CHECKING:
    from collections.abc import MutableMapping

    from agex.fs import FSConfig

    from ..fs.aware import AgentAwareFS
    from ..host import Host
    from ..host.dependencies import Dependencies
    from ..state.config import StateConfig

_UNSET = object()


def clear_agent_registry() -> None:
    """Clear agent name tracking and dynamic dataclass registry. Primarily for testing."""
    from .task import clear_dynamic_dataclass_registry

    with BaseAgent._used_names_lock:
        BaseAgent._used_names.clear()
    clear_dynamic_dataclass_registry()


def _random_name() -> str:
    return f"agent_{uuid.uuid4().hex[:8]}"


class BaseAgent:
    _used_names: ClassVar[dict[str, int]] = {}
    _used_names_lock: ClassVar[threading.Lock] = threading.Lock()

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
        # Event log chaptering (optional)
        chaptering_trigger: int | None = None,
        # Memory limit (passed to sandtrap)
        max_memory_mb: int | None = None,
        # File descriptor limit (per-task, Unix only)
        max_open_files: int | None = None,
        # Tick-based execution limit (primary runaway-code protection)
        eval_tick_limit: int | None = 100_000,
        # Sandbox isolation level (passed to sandtrap)
        isolation: Isolation = "none",
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
        self.eval_tick_limit = eval_tick_limit
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

        # Event log chaptering settings
        self.chaptering_trigger = chaptering_trigger

        # Memory limit passed through to sandtrap's Policy.memory_limit
        self.max_memory_mb = max_memory_mb

        # Sandbox isolation level (passed to sandtrap's sandbox() factory)
        self.isolation: Isolation = isolation

        # File descriptor limits (per-task, Unix only)
        self._resource_limits = ResourceLimits(
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

        # Skill files (name → {relative_path: content_bytes})
        self._skills: list[tuple[str, dict[str, bytes]]] = []

        # Custom terminal commands (name → registration record).
        # Populated by ``agent.terminal(...)`` (public) and
        # ``agent._terminal_command_factory(...)`` (internal, used by
        # ``register_git``); consumed by ``build_terminal_commands``
        # to construct the per-action commands dict handed to termish.
        from agex.terminal import TerminalCommandRegistration

        self._terminal_commands: dict[str, TerminalCommandRegistration] = {}

        # Fingerprint (lazy — computed on first access)
        self._fingerprint: str | None = None

        # Enforce unique agent names
        with BaseAgent._used_names_lock:
            if self.name in BaseAgent._used_names:
                existing_id = BaseAgent._used_names[self.name]
                if existing_id != id(self):
                    raise ValueError(f"Agent name '{self.name}' already exists")
            BaseAgent._used_names[self.name] = id(self)

    @property
    def fingerprint(self) -> str:
        """Agent fingerprint, computed lazily from policy configuration."""
        if self._fingerprint is None:
            self._fingerprint = compute_agent_fingerprint_from_policy(self)
        return self._fingerprint

    @fingerprint.setter
    def fingerprint(self, value: str | None) -> None:
        self._fingerprint = value

    def _update_fingerprint(self):
        """Invalidate fingerprint and dependency caches after registration changes."""
        self._cached_dependencies = None
        self._fingerprint = None

    def __getstate__(self) -> dict[str, Any]:
        """
        Custom pickling state to handle runtime objects.

        Excludes:
        - _host_object_registry: Holds live instances (db connections, etc.)
        - llm: Live LLM has nonserializable state (sockets, SSL context)
        - _host: Host has non-serializable state (HTTP clients, etc.)
        - _fingerprint: Computed from runtime state, might differ on host

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
        state.pop("_fingerprint", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore agent from pickle state.

        NOTE: The agent is not fully functional until rehydrated by prepare_agent,
        which must:
        1. Reconstruct the LLM from _llm_config
        2. Reconstruct the Host from _host_config
        """
        # Restore configuration
        self.__dict__.update(state)

        # Initialize runtime fields
        self._host_object_registry = {}  # Empty on new host

        # llm and _host remain None until rehydrated by prepare_agent
        self.llm = None
        self._host = None
        self._fingerprint = None  # Recomputes lazily on first access

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

    @staticmethod
    def _slugify(name: str) -> str:
        """Coerce a name into a path-safe slug."""
        import re

        slug = name.lower()
        slug = re.sub(r"[^a-z0-9]+", "-", slug)
        return slug.strip("-") or "skill"

    @staticmethod
    def _parse_skill_name(content: bytes, fallback: str) -> str:
        """Extract the name from YAML frontmatter, or use fallback."""
        import re

        text = content.decode("utf-8", errors="replace")
        fm_match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
        if fm_match:
            for line in fm_match.group(1).splitlines():
                line = line.strip()
                if line.startswith("name:"):
                    name = line[5:].strip().strip("\"'")
                    if name:
                        return name
        return fallback

    @staticmethod
    def _walk_skill_dir(root: Any) -> dict[str, bytes]:
        """Recursively read all files under *root*, returning {relative_posix_path: bytes}.

        Works with both pathlib.Path and importlib.resources Traversable objects.
        Skips dotfiles and dotdirectories (e.g. .git, .DS_Store).
        """
        files: dict[str, bytes] = {}

        def _walk(node: Any, prefix: str) -> None:
            for child in node.iterdir():
                child_name = getattr(child, "name", None) or ""
                if child_name.startswith("."):
                    continue
                rel = (
                    f"{prefix}{child_name}" if not prefix else f"{prefix}/{child_name}"
                )
                if hasattr(child, "iterdir") and child.is_dir():
                    _walk(child, rel)
                elif hasattr(child, "read_bytes"):
                    files[rel] = child.read_bytes()

        _walk(root, "")
        return files

    def skill(self, source: Any) -> None:
        """Register a skill for the agent.

        Skills are mounted read-only at /skills/<name>/. A skill can be a single
        file (mounted as SKILL.md) or a directory containing SKILL.md and any
        number of sibling documents.

        The skill name is taken from YAML frontmatter in SKILL.md if present,
        otherwise derived from the filename or parent directory.

        Args:
            source: Path-like object (file or directory), importlib Traversable,
                    or raw bytes.

        Examples:
            from importlib.resources import files
            agent.skill(files("calgebra") / "skills" / "calgebra" / "SKILL.md")
            agent.skill(files("my_dsl") / "skills" / "my-dsl")   # directory
            agent.skill(Path("./skills/my-dsl/"))                 # directory
            agent.skill(Path("./my-skill.md"))
            agent.skill(b"---\\nname: foo\\n---\\n# Foo\\n...")
        """
        if isinstance(source, bytes):
            content = source
            fallback = "skill"
            name = self._slugify(self._parse_skill_name(content, fallback))
            self._skills.append((name, {"SKILL.md": content}))
        elif hasattr(source, "iterdir") and source.is_dir():
            # Directory source — walk the tree
            files_dict = self._walk_skill_dir(source)
            if "SKILL.md" not in files_dict:
                raise ValueError("Directory skill requires a SKILL.md file at the root")
            fallback = getattr(source, "name", None) or "skill"
            name = self._slugify(
                self._parse_skill_name(files_dict["SKILL.md"], fallback)
            )
            self._skills.append((name, files_dict))
        elif hasattr(source, "read_bytes"):
            content = source.read_bytes()
            filename = getattr(source, "name", None) or "skill"
            if filename == "SKILL.md" and hasattr(source, "parent"):
                parent_name = getattr(source.parent, "name", None)
                fallback = parent_name or "skill"
            else:
                fallback = os.path.splitext(filename)[0]
            name = self._slugify(self._parse_skill_name(content, fallback))
            self._skills.append((name, {"SKILL.md": content}))
        else:
            raise TypeError(
                f"skill() expects bytes, a Path-like object, or a directory, "
                f"got {type(source).__name__}"
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

    @property
    def scope_names(self) -> set[str]:
        """The capability scopes declared across this agent's registrations
        (the scopes that *can* be granted).

        Static — derived from the registration set, not from any session's
        grant state. Distinct from ``scopes(state)`` which reports the scopes
        currently *granted* in a given session.
        """
        names: set[str] = set()
        for ns in self._policy.namespaces.values():
            if ns.scope:
                names.add(ns.scope)
            for spec in ns.fns.values():
                if getattr(spec, "scope", None):
                    names.add(spec.scope)  # type: ignore[arg-type]
            for rc in ns.classes.values():
                if getattr(rc, "scope", None):
                    names.add(rc.scope)  # type: ignore[arg-type]
        return names

    def state(self, session: str = "default") -> "MutableMapping[str, Any]":
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
                fs=connect_fs(type="isolated", root="/project"),
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

        from agex.fs import IsolatedFS, IsolatedFSConfig, VirtualFS, VirtualFSConfig

        state = self.state(session)

        if isinstance(self._fs_config, VirtualFSConfig):
            return VirtualFS(state, max_size_mb=self._fs_config.max_size_mb), state

        elif isinstance(self._fs_config, IsolatedFSConfig):
            from agex.eval.core import _get_session_root

            # Get session-specific root if per_session is enabled
            per_session = self._fs_config.per_session
            root = _get_session_root(self._fs_config.root, session, per_session)

            return IsolatedFS(root), state

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
