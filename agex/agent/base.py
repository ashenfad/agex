import uuid
from contextvars import ContextVar
from typing import Any, Callable, Dict, Literal

from ..llm import LLMClient, connect_llm
from .datatypes import (
    MemberSpec,
    RegisteredClass,
)
from .fingerprint import compute_agent_fingerprint_from_policy
from .policy.policy import AgentPolicy

# Global registry mapping fingerprints to agents
# Using ContextVar for thread/async-task safety in server environments
_AGENT_REGISTRY: ContextVar[Dict[str, "BaseAgent"]] = ContextVar(
    "agent_registry", default={}
)
# Global registry mapping agent names to agents
_AGENT_REGISTRY_BY_NAME: ContextVar[Dict[str, "BaseAgent"]] = ContextVar(
    "agent_registry_by_name", default={}
)


def register_agent(agent: "BaseAgent") -> str:
    """
    Register an agent in the global registry.

    Returns the agent's fingerprint.
    """
    registry_by_name = _AGENT_REGISTRY_BY_NAME.get().copy()
    registry = _AGENT_REGISTRY.get().copy()

    # Enforce unique agent names if provided
    if hasattr(agent, "name") and agent.name is not None:
        if agent.name in registry_by_name:
            existing_agent = registry_by_name[agent.name]
            if existing_agent is not agent:  # Allow re-registration of same agent
                raise ValueError(f"Agent name '{agent.name}' already exists")
        registry_by_name[agent.name] = agent
        _AGENT_REGISTRY_BY_NAME.set(registry_by_name)

    fingerprint = compute_agent_fingerprint_from_policy(agent)
    registry[fingerprint] = agent
    _AGENT_REGISTRY.set(registry)
    return fingerprint


def resolve_agent(fingerprint: str) -> "BaseAgent":
    """
    Resolve an agent by its fingerprint.

    Raises RuntimeError if no matching agent is found.
    """
    registry = _AGENT_REGISTRY.get()
    agent = registry.get(fingerprint)
    if not agent:
        available = list(registry.keys())
        raise RuntimeError(
            f"No agent found with fingerprint '{fingerprint[:8]}...'. "
            f"Available fingerprints: {[fp[:8] + '...' for fp in available]}"
        )
    return agent


def clear_agent_registry() -> None:
    """Clear the global registry. Primarily for testing."""
    from .task import clear_dynamic_dataclass_registry

    _AGENT_REGISTRY.set({})
    _AGENT_REGISTRY_BY_NAME.set({})
    clear_dynamic_dataclass_registry()


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
        llm_client: LLMClient | None = None,
        # LLM retry control (timeout comes from llm_client.timeout_seconds)
        llm_max_retries: int = 2,
        # Event log summarization (optional)
        log_high_water_tokens: int | None = None,
        log_low_water_tokens: int | None = None,
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

        # Create LLM client using the resolved configuration
        self.llm_client = llm_client or connect_llm()
        # LLM retry setting (timeout comes from llm_client.timeout_seconds)
        self.llm_max_retries = llm_max_retries

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

        # private, host-side registry for live, unpickleable objects
        self._host_object_registry: dict[str, Any] = {}

        self._policy: AgentPolicy = AgentPolicy()

        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self.fingerprint = register_agent(self)

    def __getstate__(self) -> dict[str, Any]:
        """
        Custom pickling state to handle runtime objects.

        Excludes:
        - _host_object_registry: Holds live instances (db connections, etc.)
        - llm_client: Live client has nonserializable state (sockets, SSL context)
        - fingerprint: Computed from runtime state, might differ on host

        Adds:
        - _llm_config: Reconstructable configuration for the LLM client
        """
        state = self.__dict__.copy()

        # Serialize LLM config using our new helper
        if self.llm_client:
            state["_llm_config"] = self.llm_client.dump_config()

        # Remove runtime-only objects
        state.pop("llm_client", None)
        state.pop("_host_object_registry", None)
        state.pop("fingerprint", None)

        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore agent from pickle state.

        NOTE: The agent is not fully functional until rehydrated by the remote
        runtime, which must:
        1. Inject a new llm_client (or use the one from _llm_config)
        2. Recompute fingerprint/register
        """
        # Restore configuration
        self.__dict__.update(state)

        # Initialize runtime fields that were mocked/missing
        self._host_object_registry = {}  # Empty on new host

        # llm_client remains None until injected by deserialize_agent
        # or lazily connected if we want that behavior (design choice: passed in)
        self.llm_client = None
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
