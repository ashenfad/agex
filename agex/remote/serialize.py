from typing import Optional

import cloudpickle

from agex.agent import Agent
from agex.llm import LLMClient


def serialize_agent(agent: Agent) -> bytes:
    """
    Serialize an agent and its dependencies to bytes using cloudpickle.

    This captures:
    - The agent configuration and policy
    - The task function and its closure (dependent variables)
    - Registered modules (as imports)
    - Dependent agents (via closure capture)
    - LLM configuration (via __getstate__)

    It excludes:
    - Host-specific objects (DB connections, valid only on origin)
    - The live LLM client instance
    """
    return cloudpickle.dumps(agent)


def deserialize_agent(payload: bytes, llm_client: Optional[LLMClient] = None) -> Agent:
    """
    Deserialize an agent from bytes.

    Args:
        payload: Pickled agent bytes
        llm_client: Optional pre-configured LLM client to inject. If not provided,
                   will attempt to reconstruct from serialized config.

    Returns:
        Rehydrated Agent instance, ready for execution.

    Raises:
        ValueError: If payload is not a valid Agent
    """
    agent = cloudpickle.loads(payload)

    if not isinstance(agent, Agent):
        raise ValueError(f"Deserialized object is not an Agent: {type(agent)}")

    # Rehydrate LLM client
    if llm_client:
        agent.llm_client = llm_client
    elif hasattr(agent, "_llm_config") and agent._llm_config:
        try:
            # Reconstruct from config using the LLMClient factory
            # avoiding circular import at top level
            from agex.llm import LLMClient

            agent.llm_client = LLMClient.from_config(agent._llm_config)
        except Exception as e:
            # Fallback or strict failure?
            # For now, let's allow it but warn, or maybe fail if strict.
            # Design choice: If we can't create the LLM client, the agent is useless.
            raise RuntimeError(
                f"Failed to reconstruct LLM client from config: {e}"
            ) from e
    else:
        # No client provided and no config found.
        # This might be okay if the agent doesn't need LLM (unlikely for Agex)
        # or if it relies on default env vars on the remote host.
        from agex.llm import connect_llm

        agent.llm_client = connect_llm()

    # Re-register the agent in the new process global registry
    # This recomputes the fingerprint based on the restored state
    agent._update_fingerprint()

    return agent
