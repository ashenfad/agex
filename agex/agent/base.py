from typing import Dict

from ..llm import get_llm_client
from ..llm.config import get_llm_config
from .datatypes import (
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
)
from .fingerprint import compute_agent_fingerprint

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

    fingerprint = compute_agent_fingerprint(
        agent.primer, agent.fn_registry, agent.cls_registry, agent.importable_modules
    )
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
    global _AGENT_REGISTRY, _AGENT_REGISTRY_BY_NAME
    _AGENT_REGISTRY = {}
    _AGENT_REGISTRY_BY_NAME = {}


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
        llm_provider: str | None = None,
        llm_model: str | None = None,
        **llm_kwargs,
    ):
        self.name = name
        self.primer = primer
        self.timeout_seconds = timeout_seconds
        self.max_iterations = max_iterations
        self.max_tokens = max_tokens

        # Get smart LLM configuration with fallback chain
        self.llm_config = get_llm_config(
            provider=llm_provider, model=llm_model, **llm_kwargs
        )

        # Create LLM client using the resolved configuration
        self.llm_client = get_llm_client(**self.llm_config)

        # Agent registries
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.cls_registry_by_type: dict[type, RegisteredClass] = {}
        self.importable_modules: dict[str, RegisteredModule] = {}

        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self.fingerprint = register_agent(self)
