# Main agent functionality
from .core import Agent, clear_agent_registry, register_agent, resolve_agent

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
    _AgentExit,
    task,
)

# Fingerprinting (usually internal, but exported for testing)
from .fingerprint import compute_agent_fingerprint

__all__ = [
    # Core functionality
    "Agent",
    "register_agent",
    "resolve_agent",
    "clear_agent_registry",
    # Agent exit system
    "_AgentExit",
    "ExitSuccess",
    "ExitFail",
    "ExitClarify",
    "Task",
    "task",
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
    "compute_agent_fingerprint",
]
