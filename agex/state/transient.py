"""
A state wrapper that allows transient (unpickleable) variables in a controlled scope.

This enables patterns like `with obj as var:` where obj might return unpickleable
intermediate values, bypassing pickle safety checks for variables marked as transient.
"""

from typing import Any, Iterable

from .core import State
from .live import Live


class TransientScope(State):
    """
    A state manager that allows certain variables to be transient (unpickleable).

    Transient variables are stored in a separate store without pickle safety checks.
    All other variables follow normal safety rules. This enables temporary use of
    unpickleable objects like database cursors within a controlled scope.
    """

    def __init__(self, parent_store: State, transient_vars: set[str] | None = None):
        self._local_store = Live()
        self._transient_store = {}  # Raw dict for unpickleable objects
        self._parent_store = parent_store
        self._transient_vars = transient_vars or set()
        super().__init__()

    @property
    def base_store(self) -> "State":
        return self._parent_store

    def get(self, key: str, default: Any = None) -> Any:
        # Check transient store first (for unpickleable objects)
        if key in self._transient_store:
            return self._transient_store[key]

        # Then check local store (for normal scoped variables)
        if key in self._local_store:
            return self._local_store.get(key, default)

        # Finally delegate to parent
        return self._parent_store.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if key in self._transient_vars:
            # Store transient variables without pickle safety checks
            self._transient_store[key] = value
        else:
            # For non-transient variables, try normal assignment first
            # If it fails due to pickle safety, make the variable transient
            try:
                self._parent_store.set(key, value)
            except Exception as e:
                if "unpickleable" in str(e).lower() or "pickle" in str(e).lower():
                    # Make this variable transient and store it
                    self._transient_vars.add(key)
                    self._transient_store[key] = value
                else:
                    # Re-raise other exceptions
                    raise

    def remove(self, key: str) -> bool:
        # Try transient store first
        if key in self._transient_store:
            del self._transient_store[key]
            return True

        # Then try local store
        return self._local_store.remove(key)

    def add_transient_var(self, key: str) -> None:
        """Mark a variable as transient (bypassing pickle safety)."""
        self._transient_vars.add(key)

    def remove_transient_var(self, key: str) -> None:
        """Remove a variable from transient tracking."""
        self._transient_vars.discard(key)
        # Also remove from transient store if present
        self._transient_store.pop(key, None)

    def keys(self) -> Iterable[str]:
        raise NotImplementedError("Not supported for transient scope.")

    def values(self) -> Iterable[Any]:
        raise NotImplementedError("Not supported for transient scope.")

    def items(self) -> Iterable[tuple[str, Any]]:
        raise NotImplementedError("Not supported for transient scope.")

    def __contains__(self, key: str) -> bool:
        return (
            key in self._transient_store
            or key in self._local_store
            or key in self._parent_store
        )
