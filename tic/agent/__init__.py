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
        super().__init__(primer, timeout_seconds)

    def task(self, func):
        """A decorator to mark a function as an agent task."""
        from .datatypes import Task

        return Task()
