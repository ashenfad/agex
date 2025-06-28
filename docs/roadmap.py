"""
TIC Library Roadmap
==================

This document outlines planned improvements and architectural decisions for the tic library.
Since typing is central to capturing agent contracts, many items focus on type system improvements.

PRIORITY LEVELS:
- P0: Critical/blocking issues
- P1: High impact, should do soon
- P2: Medium impact, nice to have
- P3: Low priority, future consideration

"""

# =============================================================================
# TYPING SYSTEM IMPROVEMENTS
# =============================================================================

# P1: ParamSpec Adoption for Task Decorators
# ------------------------------------------
# GOAL: Replace `# type: ignore[call-arg]` with proper ParamSpec typing
# BLOCKED BY: mypy's incomplete PEP-612 support
# DECISION: Adopt when mypy support improves (likely 2024-2025)
#
# BENEFITS:
# - Eliminates need for type: ignore comments
# - Proper type checking for task decorator state injection
# - Better IDE experience for users with Pyright/Pylance
# - Future-proof typing architecture
#
# IMPLEMENTATION PLAN:
# 1. Monitor mypy PEP-612 progress (issues #8645, #11833, #12011)
# 2. Create experimental branch with ParamSpec implementation
# 3. Test with major type checkers
# 4. Migrate when mypy support is stable
#
# MIGRATION STRATEGY:
# - Provide fallback for older mypy versions
# - Document type checker recommendations
# - Update examples to use ParamSpec patterns

TODO_PARAMSPEC_ADOPTION = """
from typing import ParamSpec, TypeVar
from agex.agent.datatypes import StateType

P = ParamSpec('P')
R = TypeVar('R')

class TaskMixin:
    def task(self, func: Callable[P, R]) -> Callable[Concatenate[StateType, P], R]:
        # Proper ParamSpec implementation without type: ignore
        ...
"""

# P2: Enhanced Type Safety for Agent Contracts
# --------------------------------------------
# GOAL: Stronger typing for agent state and communication
#
# AREAS FOR IMPROVEMENT:
# - State type validation at runtime
# - Better type inference for agent responses
# - Stricter typing for module registration
# - Protocol-based interfaces for extensibility

TODO_AGENT_CONTRACT_TYPING = """
# Potential Protocol-based agent interface
from typing import Protocol, runtime_checkable

@runtime_checkable
class AgentProtocol(Protocol[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...
    def register_module(self, module: ModuleType, **config) -> None: ...
"""

# P2: Type-Safe Module Registration
# ---------------------------------
# GOAL: Better typing for agent.module() and agent.cls() decorators
#
# CURRENT ISSUES:
# - Module registration accepts Any for configure parameter
# - Class registration has loose type bounds
# - Limited compile-time validation of visibility settings
#
# IMPROVEMENTS:
# - TypedDict for configure parameters
# - Literal types for visibility levels
# - Better integration with MemberSpec

TODO_MODULE_REGISTRATION_TYPING = """
from typing import TypedDict, Literal, NotRequired

class ModuleConfig(TypedDict, total=False):
    visibility: NotRequired[Literal["high", "medium", "low"]]
    configure: NotRequired[Dict[str, MemberSpec]]

class AgentMixin:
    def module(self, 
               module: ModuleType, 
               *,
               visibility: Literal["high", "medium", "low"] = "medium",
               configure: ModuleConfig | None = None) -> None: ...
"""

# =============================================================================
# ARCHITECTURE & API IMPROVEMENTS
# =============================================================================

# P1: Task Decorator Interface Redesign
# -------------------------------------
# STATUS: ✅ COMPLETED (required primer parameter with caller-facing docstrings)
#
# COMPLETED:
# - Changed @agent.task to require primer parameter: @agent.task("instructions")
# - Primer = agent-facing instructions (how to implement)
# - Docstring = developer-facing documentation (how to call/use)
# - Updated all examples and tests to follow new pattern
# - Enables dual-decorated functions for multi-agent coordination
# IMPACT: Clear separation of concerns, better multi-agent patterns

# P1: Consistent Visibility System
# --------------------------------
# STATUS: ✅ COMPLETED (fixed _should_render_member logic)
#
# COMPLETED:
# - Fixed medium visibility functions to render properly
# - Made class attributes consistent with methods
# - All medium-visibility items now show signatures without docstrings

# P2: Enhanced State Management
# ----------------------------
# GOAL: Better abstractions for agent state persistence and versioning
#
# CURRENT STATE: Basic Versioned/Ephemeral/Scoped state classes exist
# IMPROVEMENTS NEEDED:
# - Better serialization story
# - State migration utilities
# - More ergonomic state access patterns
# - Transaction-like state operations

TODO_STATE_MANAGEMENT = """
# Potential improvements to state system
class Agent:
    @contextmanager
    def state_transaction(self):
        # Atomic state operations
        pass
    
    def migrate_state(self, old_version: str, new_version: str):
        # Schema migration utilities
        pass
"""

# P3: Performance Optimizations
# -----------------------------
# GOAL: Optimize hot paths in evaluation and rendering
#
# AREAS TO INVESTIGATE:
# - Caching of rendered definitions
# - Lazy evaluation improvements
# - State access optimization
# - Memory usage in long-running agents

# P3: Enhanced Error Handling
# ---------------------------
# GOAL: Better error messages and debugging experience
#
# IMPROVEMENTS:
# - Source location tracking in evaluation errors
# - Better error context for state issues
# - Enhanced debugging utilities for agent development

# =============================================================================
# ECOSYSTEM & DEVELOPER EXPERIENCE
# =============================================================================

# P2: Documentation Improvements
# ------------------------------
# GOAL: Comprehensive documentation for agent development
#
# NEEDED:
# - Type checker setup guide (mypy vs Pyright)
# - Advanced agent patterns cookbook
# - Migration guide for ParamSpec adoption (when ready)
# - Performance tuning guide

# P2: Testing Infrastructure
# --------------------------
# GOAL: Better testing utilities for agent development
#
# IMPROVEMENTS:
# - Agent testing fixtures
# - State mocking utilities
# - Type checking test helpers
# - Integration test patterns

# P3: IDE Integration
# ------------------
# GOAL: Enhanced development experience
#
# POTENTIAL:
# - Language server integration
# - Agent debugging extensions
# - State inspection tools

# =============================================================================
# EXTERNAL DEPENDENCIES & MONITORING
# =============================================================================

# Monitor: mypy PEP-612 Support
# -----------------------------
# TRACK: https://github.com/python/mypy/issues/8645
# DECISION POINT: When to adopt ParamSpec
#
# CRITERIA FOR ADOPTION:
# - Basic ParamSpec support stable in mypy
# - Concatenate support working
# - No crashes on common usage patterns
# - Reasonable error messages

# Monitor: Python Type System Evolution
# ------------------------------------
# RELEVANT PEPS:
# - PEP 695: Type Parameter Syntax (Python 3.12+)
# - PEP 718: Subscriptable functions (proposed)
# - Future generic improvements
#
# IMPACT: May provide alternative solutions to current typing challenges

if __name__ == "__main__":
    print("TIC Library Roadmap - see source for detailed planning")
    print("Priority items:")
    print("- P1: ParamSpec adoption (blocked by mypy)")
    print("- P1: ✅ Task decorator interface redesign (completed)")
    print("- P1: ✅ Visibility system fixes (completed)")
    print("- P2: Enhanced agent contract typing")
    print("- P2: Type-safe module registration")
