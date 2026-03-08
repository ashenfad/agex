"""State configuration for agent state management."""

from dataclasses import dataclass
from typing import Any, Callable, Literal

# Type alias for init parameter
InitVars = Callable[[], dict[str, Any]] | dict[str, Any] | None


@dataclass
class StateConfig:
    """
    Configuration for state management.

    This describes HOW state works (type, storage, GC params).
    The actual State instance is created by the Host at execution time.

    Args:
        type: State semantics ("ephemeral", "versioned", or "live")
        storage: Storage backend ("memory" or "disk")
        path: Directory path for disk storage
        high_water_bytes: Trigger GC when total size exceeds this (versioned only)
        low_water_bytes: Target size after GC (versioned only, default: 80% of high_water)
        init: Callable or dict to initialize state variables on first session creation
    """

    type: Literal["ephemeral", "versioned", "live"]
    storage: Literal["memory", "disk", "indexeddb"] | None = None
    path: str | None = None
    high_water_bytes: int | None = None
    low_water_bytes: int | None = None
    options: dict[str, Any] | None = None
    init: InitVars = None

    def dump_config(self) -> dict[str, Any]:
        """Serialize for remote reconstruction."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StateConfig":
        """Reconstruct from config dict."""
        return cls(**config)
