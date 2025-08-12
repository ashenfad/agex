# Main agent functionality
from ..llm import LLMClient
from .base import BaseAgent, clear_agent_registry, register_agent, resolve_agent

# Data types and exceptions
from .datatypes import (
    RESERVED_NAMES,
    AttrDescriptor,
    MemberSpec,
    Pattern,
    RegisteredClass,
    RegisteredFn,
    RegisteredItem,
    RegisteredModule,
    TaskContinue,
    TaskFail,
    TaskSuccess,
    Visibility,
)
from .loop import TaskLoopMixin

# Fingerprinting (usually internal, but exported for testing)
from .registration import RegistrationMixin
from .task import TaskMixin, clear_dynamic_dataclass_registry

__all__ = [
    # Core functionality
    "register_agent",
    "resolve_agent",
    "clear_agent_registry",
    "clear_dynamic_dataclass_registry",
    # Task control functions
    "TaskSuccess",
    "TaskFail",
    "TaskContinue",
    # Registration types
    "MemberSpec",
    "AttrDescriptor",
    "RegisteredItem",
    "RegisteredFn",
    "RegisteredClass",
    "RegisteredModule",
    # Type aliases and constants
    "Pattern",
    "Visibility",
    "RESERVED_NAMES",
    # Fingerprinting
]


class Agent(RegistrationMixin, TaskMixin, TaskLoopMixin, BaseAgent):
    def __init__(
        self,
        primer: str | None = None,
        timeout_seconds: float = 5.0,
        max_iterations: int = 10,
        max_tokens: int = 2**16,
        # Agent identification
        name: str | None = None,
        # LLM configuration (optional, uses smart defaults)
        llm_client: LLMClient | None = None,
        # LLM retry controls
        llm_max_retries: int = 2,
        llm_retry_backoff: float = 0.25,
    ):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            timeout_seconds: The maximum time in seconds for a single action evaluation.
            max_iterations: The maximum number of think-act cycles for a task.
            max_tokens: The maximum number of tokens to use for context rendering.
            name: Unique identifier for this agent (for sub-agent namespacing).
            llm_client: An instantiated LLMClient for the agent to use.
        """
        super().__init__(
            primer,
            timeout_seconds,
            max_iterations,
            max_tokens,
            name=name,
            llm_client=llm_client,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
        )
