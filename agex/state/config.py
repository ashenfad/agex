"""State configuration for agent state management."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

if TYPE_CHECKING:
    from agex.state.resolver import StateResolver

# Type alias for init parameter
InitVars = Callable[[], dict[str, Any]] | dict[str, Any] | None


@dataclass
class StateConfig:
    """
    Configuration for state management.

    This describes HOW state works (type, storage, backend options).
    The actual State instance is created by the Host at execution time.

    Args:
        type: State semantics ("ephemeral", "versioned", "live", or "resolver")
        storage: Storage backend ("memory", "disk", or "indexeddb")
        path: Directory path for disk storage
        init: Callable or dict to initialize state variables on first session creation
        resolver: Bring-your-own per-session state lookup (type="resolver" only)
    """

    type: Literal["ephemeral", "versioned", "live", "resolver"]
    storage: Literal["memory", "disk", "indexeddb"] | None = None
    path: str | None = None
    options: dict[str, Any] | None = None
    init: InitVars = None
    resolver: "StateResolver | None" = None

    def dump_config(self) -> dict[str, Any]:
        """Serialize for remote reconstruction."""
        if self.type == "resolver":
            # A resolver is a live in-process object — it can't travel to a
            # remote host. Fail loudly rather than serializing a stub.
            raise ValueError(
                "StateConfig with a custom resolver cannot be serialized for "
                "remote execution; resolvers are Local-host only"
            )
        return {
            k: v for k, v in self.__dict__.items() if v is not None and k != "resolver"
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "StateConfig":
        """Reconstruct from config dict."""
        return cls(**config)
