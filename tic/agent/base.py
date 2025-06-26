from typing import Dict

from .datatypes import (
    RegisteredClass,
    RegisteredFn,
    RegisteredModule,
)
from .fingerprint import compute_agent_fingerprint

# Global registry mapping fingerprints to agents
_AGENT_REGISTRY: Dict[str, "BaseAgent"] = {}


def register_agent(agent: "BaseAgent") -> str:
    """
    Register an agent in the global registry.

    Returns the agent's fingerprint.
    """
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
    global _AGENT_REGISTRY
    _AGENT_REGISTRY = {}


class BaseAgent:
    def __init__(self, primer: str | None = None, timeout_seconds: float = 5.0):
        self.primer = primer
        self.timeout_seconds = timeout_seconds
        self.fn_registry: dict[str, RegisteredFn] = {}
        self.cls_registry: dict[str, RegisteredClass] = {}
        self.cls_registry_by_type: dict[type, RegisteredClass] = {}
        self.importable_modules: dict[str, RegisteredModule] = {}
        # Auto-register this agent
        self.fingerprint = register_agent(self)

    def _update_fingerprint(self):
        """Update the fingerprint after registration changes."""
        self.fingerprint = register_agent(self)
