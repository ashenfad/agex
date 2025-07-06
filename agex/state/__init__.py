"""
A state management system for tic agents.
"""

from .core import State, is_ephemeral_root
from .ephemeral import Ephemeral
from .kv import KVStore
from .namespaced import Namespaced
from .scoped import Scoped
from .transient import TransientScope
from .versioned import Versioned

__all__ = [
    "State",
    "is_ephemeral_root",
    "Ephemeral",
    "KVStore",
    "Namespaced",
    "Scoped",
    "TransientScope",
    "Versioned",
]
