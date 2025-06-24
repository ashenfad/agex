from typing import Any

from ..eval.objects import PrintTuple
from ..state import State


def dir_builtin(obj: Any = None, *, state: "State") -> None:
    """
    A custom implementation of 'dir'.
    If called with no arguments, lists names in the current scope.
    If called with an object, lists its public attributes.
    This is a side-effect function that writes to __stdout__ in the state.
    """
    result: list[str]
    if obj is None:
        # Get all keys from the current state scope.
        keys = state.keys()
        result = sorted(list(keys))
    else:
        # Get attributes and filter out private/magic ones.
        result = sorted([attr for attr in dir(obj) if not attr.startswith("_")])

    # Now, write the generated list to stdout.
    current_stdout = state.get("__stdout__")
    if not isinstance(current_stdout, list):
        current_stdout = []

    # The real dir() prints the list itself.
    new_stdout = current_stdout + [PrintTuple([result])]
    state.set("__stdout__", new_stdout)
