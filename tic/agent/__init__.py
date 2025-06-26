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

# Fingerprinting (usually internal, but exported for testing)
from .registration import RegistrationMixin
from .task_loop import TaskLoopMixin

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


class Agent(RegistrationMixin, TaskLoopMixin, BaseAgent):
    def __init__(self, primer: str | None = None, timeout_seconds: float = 5.0):
        """
        An agent that can be used to execute tasks.

        Args:
            primer: A string to guide the agent's behavior.
            timeout_seconds: The maximum time in seconds to execute a task.
        """
        super().__init__(primer, timeout_seconds)
