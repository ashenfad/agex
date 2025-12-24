"""
Serialization utilities for remote host execution.

This module handles the serialization and deserialization of agents for
transport between client and host.
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

    The agent's LLM is reconstructed from the serialized configuration.
    The server environment must have the appropriate API keys set (e.g.,
    ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY).

    Args:
        payload: Pickled bytes from serialize_agent

    Returns:
        Reconstructed Agent instance

    Raises:
        ValueError: If payload is not a valid Agent
        RuntimeError: If LLM cannot be reconstructed
        ImportError: If cloudpickle is not available
    """
    cloudpickle = _get_cloudpickle()
    agent = cloudpickle.loads(payload)

    # Duck typing: check for agent-specific attributes to avoid circular import
    if not _is_agent_like(agent):
        raise ValueError(f"Deserialized object is not an Agent: {type(agent)}")

    # Rehydrate LLM from serialized config
    if hasattr(agent, "_llm_config") and agent._llm_config:
        try:
            from agex.llm import LLM

            agent.llm = LLM.from_config(agent._llm_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to reconstruct LLM from config: {e}. "
                f"Ensure the server has the appropriate API keys set."
            ) from e
    else:
        raise RuntimeError(
            "Agent has no LLM configuration. Ensure the agent has an "
            "llm set before serialization."
        )

    # Re-register the agent in the new process global registry
    agent._update_fingerprint()

    return agent


def _is_agent_like(obj: Any) -> bool:
    """
    Check if an object looks like an Agent using duck typing.

    This avoids importing Agent which would cause circular imports.
    """
    required_attrs = ("_tasks", "fingerprint", "_update_fingerprint", "llm", "name")
    return all(hasattr(obj, attr) for attr in required_attrs)
