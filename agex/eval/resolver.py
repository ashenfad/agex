from __future__ import annotations

from typing import Any

from .builtins import BUILTINS
from .error import EvalError
from .objects import AgexInstance, AgexModule, AgexObject, BoundInstanceObject
from .user_errors import AgexAttributeError
from .utils import get_allowed_attributes_for_instance


class Resolver:
    """
    Facade over the agent's eager registries used by the evaluator.

    This centralizes name, attribute, and import resolution so we can later
    swap it with a lazy policy-based implementation without touching evaluator
    logic.
    """

    def __init__(self, agent):
        self.agent = agent

    # --- Name Resolution ---
    def resolve_name(self, name: str, state, node) -> Any:
        # 1. Builtins
        if name in BUILTINS:
            return BUILTINS[name]

        # 2. State
        value = state.get(name)
        if value is not None or name in state:
            return value

        # 3. Registered live objects
        if name in self.agent.object_registry:
            reg_object = self.agent.object_registry[name]
            from .objects import BoundInstanceObject

            return BoundInstanceObject(
                reg_object=reg_object, host_registry=self.agent._host_object_registry
            )

        # 4. Registered functions
        if name in self.agent.fn_registry:
            # Local import to avoid circular dependency during module import
            from .functions import NativeFunction

            spec = self.agent.fn_registry[name]
            return NativeFunction(name=name, fn=spec.fn)

        # 5. Registered classes
        if name in self.agent.cls_registry:
            return self.agent.cls_registry[name].cls

        raise EvalError(f"Name '{name}' is not defined. (forgot import?)", node)

    # --- Attribute Resolution ---
    def resolve_attribute(self, value: Any, attr_name: str, node) -> Any:
        # Sandboxed AgexObjects and live objects have their own logic
        if isinstance(value, (AgexObject, AgexInstance)):
            return value.getattr(attr_name)

        # Host object proxy
        if isinstance(value, BoundInstanceObject):
            return value.getattr(attr_name)

        # AgexModule attribute access with JIT resolution
        if isinstance(value, AgexModule):
            reg_module = self.agent.importable_modules.get(value.name)
            if not reg_module:
                raise AgexAttributeError(
                    f"Module '{value.name}' is not registered", node
                )

            # Get attribute from host module
            try:
                real_attr = getattr(reg_module.module, attr_name)
            except AttributeError:
                raise AgexAttributeError(
                    f"module '{value.name}' has no attribute '{attr_name}'", node
                )

            # If attribute is a module, ensure it is explicitly registered
            import types

            if isinstance(real_attr, types.ModuleType):
                submodule_name = f"{value.name}.{attr_name}"
                found_spec = None
                for spec in self.agent.importable_modules.values():
                    if spec.module is real_attr:
                        found_spec = spec
                        break
                if not found_spec:
                    raise AgexAttributeError(
                        f"Submodule '{submodule_name}' is not allowed. ", node
                    )
                return AgexModule(
                    name=found_spec.name, agent_fingerprint=self.agent.fingerprint
                )

            return real_attr

        # Check for registered host classes and whitelisted methods on Python objects
        allowed_attrs = get_allowed_attributes_for_instance(self.agent, value)
        if attr_name in allowed_attrs:
            try:
                return getattr(value, attr_name)
            except AttributeError:
                raise AgexAttributeError(
                    f"'{type(value).__name__}' object has no attribute '{attr_name}'",
                    node,
                )

        raise AgexAttributeError(
            f"'{type(value).__name__}' object has no attribute '{attr_name}'", node
        )

    # --- Import Resolution ---
    def resolve_module(self, module_name: str, node) -> AgexModule:
        reg_module = self.agent.importable_modules.get(module_name)
        if not reg_module:
            raise EvalError(
                f"Module '{module_name}' is not registered or whitelisted.", node
            )
        return AgexModule(name=module_name, agent_fingerprint=self.agent.fingerprint)

    def import_from(self, module_name: str, member_name: str, node) -> Any:
        reg_module = self.agent.importable_modules.get(module_name)
        if not reg_module:
            raise EvalError(f"No module named '{module_name}' is registered.", node)

        # Simplified import model as before
        if hasattr(reg_module.module, member_name):
            return getattr(reg_module.module, member_name)
        raise EvalError(
            f"Cannot import name '{member_name}' from module '{module_name}'.", node
        )
