"""
Serialization utilities for remote agent execution.

This module handles the serialization and deserialization of agents for
transport between client and server.
"""

import cloudpickle

from agex.agent import Agent


def serialize_agent(agent: Agent) -> bytes:
    """
    Serialize an agent for transport to a remote server.

    Args:
        agent: The agent to serialize

    Returns:
        Pickled bytes representing the agent
    """
    return cloudpickle.dumps(agent)


def deserialize_agent(payload: bytes) -> Agent:
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
    """
    agent = cloudpickle.loads(payload)

    if not isinstance(agent, Agent):
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
