"""Sync helper to auto-inject submodule references into parent namespaces."""

from types import ModuleType

from .datatypes import Namespace


def _sync_submodule_attributes(namespaces: dict[str, Namespace]) -> None:
    """Ensure parent modules can access registered submodules as attributes.

    For example, if both 'os' and 'os.path' are registered, this ensures
    that accessing 'path' on the 'os' module returns the 'os.path' module.

    This handles module aliases (e.g., os.path is actually posixpath on Unix).

    Args:
        namespaces: Dict of all registered namespaces (modified in-place)
    """
    # For each registered module namespace, check if any parent module has it as an attribute
    for child_ns_name, child_ns in list(namespaces.items()):
        if child_ns.kind != "module":
            continue

        try:
            child_mod = child_ns._ensure_module_loaded()
        except Exception:
            continue

        if not isinstance(child_mod, ModuleType):
            continue

        # Check all other module namespaces to see if they have this module as an attribute
        for parent_ns_name, parent_ns in namespaces.items():
            if parent_ns.kind != "module" or parent_ns_name == child_ns_name:
                continue

            try:
                parent_mod = parent_ns._ensure_module_loaded()
            except Exception:
                continue

            if not isinstance(parent_mod, ModuleType):
                continue

            # Check if parent module has child as an attribute
            for attr_name in dir(parent_mod):
                try:
                    attr_value = getattr(parent_mod, attr_name)
                    # If the attribute is the exact same module object, record it
                    if attr_value is child_mod:
                        parent_ns.submodules[attr_name] = child_ns_name
                        break
                except Exception:
                    continue
