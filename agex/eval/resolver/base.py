from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from agex.state.core import State


@dataclass
class ModuleSpec:
    """Description of a module's origin and how to load it."""

    name: str
    origin: Literal["policy", "vfs_file", "vfs_package", "vfs_namespace"]
    loader: "BaseLoader" = field(repr=False)
    location: str | None = None  # VFS path or None for host
    submodules: dict[str, Any] = field(default_factory=dict)


class BaseFinder(ABC):
    """Abstract base class for module discovery."""

    @abstractmethod
    def find_spec(self, fullname: str) -> ModuleSpec | None:
        """Find a module spec for the given fullname."""
        pass


class BaseLoader(ABC):
    """Abstract base class for module loading."""

    @abstractmethod
    def load(self, spec: ModuleSpec, state: "State") -> Any:
        """Load the module defined by the spec into the given state."""
        pass
