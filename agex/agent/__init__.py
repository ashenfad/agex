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
from .summarization import SummarizationError
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
    # Exceptions
    "SummarizationError",
    # Fingerprinting
]


class Agent(RegistrationMixin, TaskMixin, TaskLoopMixin, BaseAgent):
    def __init__(
        self,
        primer: str | None = None,
        timeout_seconds: float = 5.0,
        max_iterations: int = 10,
        # Agent identification
        name: str | None = None,
        # Optional curated capabilities primer
        capabilities_primer: str | None = None,
        # LLM configuration (optional, uses smart defaults)
        llm_client: LLMClient | None = None,
        # LLM retry controls
        llm_max_retries: int = 2,
        llm_retry_backoff: float = 0.25,
        # Event log summarization (optional)
        log_high_water_tokens: int | None = None,
        log_low_water_tokens: int | None = None,
    ):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            timeout_seconds: The maximum time in seconds for a single action evaluation.
            max_iterations: The maximum number of think-act cycles for a task.
            name: Unique identifier for this agent (for sub-agent namespacing).
            capabilities_primer: Optional curated capabilities primer.
            llm_client: An instantiated LLMClient for the agent to use.
            log_high_water_tokens: Trigger event log summarization when total tokens
                exceed this threshold. If None, no summarization is performed.
            log_low_water_tokens: Target token count after summarization. Defaults to
                50% of log_high_water_tokens if not specified.
        """
        super().__init__(
            primer=primer,
            timeout_seconds=timeout_seconds,
            max_iterations=max_iterations,
            name=name,
            capabilities_primer=capabilities_primer,
            llm_client=llm_client,
            llm_max_retries=llm_max_retries,
            llm_retry_backoff=llm_retry_backoff,
            log_high_water_tokens=log_high_water_tokens,
            log_low_water_tokens=log_low_water_tokens,
        )
