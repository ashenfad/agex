# Main agent functionality
from .base import BaseAgent, clear_agent_registry, register_agent, resolve_agent

# Data types and exceptions
from .datatypes import (
    RESERVED_NAMES,
    AttrDescriptor,
    ExitClarify,
    ExitFail,
    ExitSuccess,
    MemberSpec,
    Pattern,
    RegisteredClass,
    RegisteredFn,
    RegisteredItem,
    RegisteredModule,
    Task,
    Visibility,
)
from .loop import TaskLoopMixin

# Fingerprinting (usually internal, but exported for testing)
from .registration import RegistrationMixin
from .task import TaskMixin

__all__ = [
    # Core functionality
    "register_agent",
    "resolve_agent",
    "clear_agent_registry",
    # Agent exit system
    "ExitSuccess",
    "ExitFail",
    "ExitClarify",
    "Task",
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
        # LLM configuration (optional, uses smart defaults)
        llm_provider: str | None = None,
        llm_model: str | None = None,
        **llm_kwargs,
    ):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            timeout_seconds: The maximum time in seconds to execute a task.
            max_iterations: The maximum number of think-act cycles for a task.
            max_tokens: The maximum number of tokens to use for context rendering.
            llm_provider: LLM provider override (falls back to global/env config).
            llm_model: LLM model override (falls back to global/env config).
            **llm_kwargs: Additional LLM parameters (temperature, etc.).
        """
        super().__init__(
            primer,
            timeout_seconds,
            max_iterations,
            max_tokens,
            llm_provider=llm_provider,
            llm_model=llm_model,
            **llm_kwargs,
        )
