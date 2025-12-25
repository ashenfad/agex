"""
Serialization utilities for remote host execution.

This module handles the basic serialization and deserialization of agents.
For full remote execution setup, use the runner module.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agex.agent.base import BaseAgent


def _get_cloudpickle():
    """Get cloudpickle, raising a clear error if unavailable."""
    try:
        import cloudpickle

        return cloudpickle
    except ImportError as e:
        raise ImportError(
            "cloudpickle is required for remote host execution. "
            "Install it with: pip install cloudpickle"
        ) from e


def serialize_agent(agent: "BaseAgent") -> bytes:
    """
    Serialize an agent for transport to a remote host.

    Args:
        agent: The agent to serialize

    Returns:
        Pickled bytes representing the agent

    Raises:
        ImportError: If cloudpickle is not available
    """
    cloudpickle = _get_cloudpickle()
    return cloudpickle.dumps(agent)


def deserialize_agent(payload: bytes) -> "BaseAgent":
    """
    Deserialize an agent from transport bytes.

    Note: This performs basic deserialization only. For full remote execution
    setup (LLM rehydration, Local host override), use prepare_agent() from
    the runner module instead.

    Args:
        payload: Pickled bytes from serialize_agent

    Returns:
        Deserialized Agent instance (not fully prepared for execution)

    Raises:
        ValueError: If payload is not a valid Agent
        ImportError: If cloudpickle is not available
    """
    cloudpickle = _get_cloudpickle()
    agent = cloudpickle.loads(payload)

    # Duck typing: check for agent-specific attributes to avoid circular import
    if not _is_agent_like(agent):
        raise ValueError(f"Deserialized object is not an Agent: {type(agent)}")

    return agent


def _is_agent_like(obj: Any) -> bool:
    """
    Check if an object looks like an Agent using duck typing.

    This avoids importing Agent which would cause circular imports.
    """
    required_attrs = ("_tasks", "fingerprint", "_update_fingerprint", "llm", "name")
    return all(hasattr(obj, attr) for attr in required_attrs)
