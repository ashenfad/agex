import uuid
from typing import Any, Callable, Dict, Literal

from ..llm import LLMClient, connect_llm
from .datatypes import (
    MemberSpec,
    RegisteredClass,
)
from .fingerprint import compute_agent_fingerprint_from_policy
from .policy.policy import AgentPolicy

# Global registry mapping fingerprints to agents
_AGENT_REGISTRY: Dict[str, "BaseAgent"] = {}
# Global registry mapping agent names to agents
_AGENT_REGISTRY_BY_NAME: Dict[str, "BaseAgent"] = {}


def register_agent(agent: "BaseAgent") -> str:
    """
    Register an agent in the global registry.

    Returns the agent's fingerprint.
    """
    # Enforce unique agent names if provided
    if hasattr(agent, "name") and agent.name is not None:
        if agent.name in _AGENT_REGISTRY_BY_NAME:
            existing_agent = _AGENT_REGISTRY_BY_NAME[agent.name]
            if existing_agent is not agent:  # Allow re-registration of same agent
                raise ValueError(f"Agent name '{agent.name}' already exists")
        _AGENT_REGISTRY_BY_NAME[agent.name] = agent

    fingerprint = compute_agent_fingerprint_from_policy(agent)
    _AGENT_REGISTRY[fingerprint] = agent
    return fingerprint


def resolve_agent(fingerprint: str) -> "BaseAgent":
    """
    Resolve an agent by its fingerprint.

    Raises RuntimeError if no matching agent is found.
    """
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

    global _AGENT_REGISTRY, _AGENT_REGISTRY_BY_NAME
    _AGENT_REGISTRY = {}
    _AGENT_REGISTRY_BY_NAME = {}
    clear_dynamic_dataclass_registry()


def _random_name() -> str:
    return f"agent_{uuid.uuid4().hex[:8]}"


class BaseAgent:
    def __init__(
        self,
        primer: str | None,
        timeout_seconds: float,
        max_iterations: int,
        max_tokens: int,
        # Agent identification
        name: str | None = None,
        # LLM configuration (optional, uses smart defaults)
        llm_client: LLMClient | None = None,
        # LLM retry controls
        llm_max_retries: int = 2,
        llm_retry_backoff: float = 0.25,
    ):
        self.name = name or _random_name()
        self.primer = primer
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

        # Create LLM client using the resolved configuration
        self.llm_client = llm_client or connect_llm()
        # LLM retry settings
        self.llm_max_retries = llm_max_retries
        self.llm_retry_backoff = llm_retry_backoff

        # private, host-side registry for live, unpickleable objects
        self._host_object_registry: dict[str, Any] = {}

        self._policy: AgentPolicy = AgentPolicy()

        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self.fingerprint = register_agent(self)

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
