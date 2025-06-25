"""
Object freezing and rehydration for eval objects during storage operations.

This module handles the serialization-friendly transformation of eval objects
(UserFunction, TicClass, etc.) when they need to be stored in versioned state.
"""

import copy
from typing import Any, Callable, Dict, Optional, Set

from ..agent import Agent

# Type alias for freeze/rehydrate handler functions
FreezeHandler = Callable[[Any], Any]
RehydrateHandler = Callable[[Any, "Agent"], Any]


class ObjectFreezer:
    """
    Handles freezing and rehydration of eval objects for storage.

    Uses a registry pattern to avoid tight coupling between storage and eval logic.
    """

    _freeze_handlers: Dict[str, FreezeHandler] = {}
    _rehydrate_handlers: Dict[str, RehydrateHandler] = {}
    _freezing_visited: Optional[Set[int]] = (
        None  # Track visited objects during freezing
    )

    @classmethod
    def register_handler(
        cls, type_name: str, freeze_fn: FreezeHandler, rehydrate_fn: RehydrateHandler
    ) -> None:
        """Register freeze/rehydrate handlers for a specific object type."""
        cls._freeze_handlers[type_name] = freeze_fn
        cls._rehydrate_handlers[type_name] = rehydrate_fn

    @classmethod
    def freeze(cls, obj: Any) -> Any:
        """
        Freeze an object for storage by removing unpickleable references.

        Returns a storage-safe version of the object.
        """
        if obj is None:
            return obj

        # Initialize visited set if this is a top-level call
        is_top_level = cls._freezing_visited is None
        if is_top_level:
            cls._freezing_visited = set()

        try:
            return cls._freeze_recursive(obj)
        finally:
            # Clean up visited set after top-level call
            if is_top_level:
                cls._freezing_visited = None

    @classmethod
    def _freeze_recursive(cls, obj: Any) -> Any:
        """Internal recursive freeze method with circular reference detection."""
        if obj is None:
            return obj

        # Check for circular references
        obj_id = id(obj)
        if cls._freezing_visited is not None and obj_id in cls._freezing_visited:
            return {
                "__circular_ref__": True,
                "__obj_type__": obj.__class__.__name__,
                "__obj_id__": obj_id,
            }

        # Add to visited set for complex objects
        needs_tracking = (
            isinstance(obj, (list, tuple, dict))
            or obj.__class__.__name__ in cls._freeze_handlers
        )
        if needs_tracking and cls._freezing_visited is not None:
            cls._freezing_visited.add(obj_id)

        try:
            type_name = obj.__class__.__name__
            freeze_handler = cls._freeze_handlers.get(type_name)

            if freeze_handler:
                return freeze_handler(obj)

            # Handle common container types recursively
            if isinstance(obj, list):
                return [cls._freeze_recursive(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(cls._freeze_recursive(item) for item in obj)
            elif isinstance(obj, dict):
                return {key: cls._freeze_recursive(value) for key, value in obj.items()}

            # No special handling needed - return as-is
            return obj
        finally:
            # Remove from visited set when done processing this object
            if needs_tracking and cls._freezing_visited is not None:
                cls._freezing_visited.discard(obj_id)

    @classmethod
    def rehydrate(cls, obj: Any, agent: "Agent") -> Any:
        """
        Rehydrate an object after loading from storage by restoring runtime context.

        Returns the object with restored agent references and live state.
        """
        if obj is None:
            return obj

        # Handle circular reference placeholders
        if isinstance(obj, dict) and obj.get("__circular_ref__"):
            # Return None as a safe fallback for circular references
            # In a more sophisticated implementation, we could try to reconstruct the reference
            return None

        type_name = obj.__class__.__name__
        rehydrate_handler = cls._rehydrate_handlers.get(type_name)

        if rehydrate_handler:
            return rehydrate_handler(obj, agent)

        # Handle common container types recursively
        if isinstance(obj, list):
            return [cls.rehydrate(item, agent) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(cls.rehydrate(item, agent) for item in obj)
        elif isinstance(obj, dict):
            return {key: cls.rehydrate(value, agent) for key, value in obj.items()}

        # No special handling needed - return as-is
        return obj


def _freeze_user_function(user_func: Any) -> Any:
    """Freeze a UserFunction by removing agent reference and converting live closure."""
    from ..state.closure import LiveClosureState
    from ..state.ephemeral import Ephemeral
    from .functions import UserFunction

    if not isinstance(user_func, UserFunction):
        return user_func

    # Create a copy to avoid modifying the original
    frozen_func = copy.copy(user_func)
    frozen_func.agent = None  # Remove agent reference to make it pickleable

    # If it has a LiveClosureState, freeze it to static
    if hasattr(frozen_func, "closure_state") and isinstance(
        frozen_func.closure_state, LiveClosureState
    ):
        # Resolve the live closure into a static, ephemeral one
        frozen_closure = Ephemeral()
        for k, v in frozen_func.closure_state.items():
            # Recursively freeze closure variables that might contain unpickleable objects
            frozen_value = ObjectFreezer._freeze_recursive(v)
            frozen_closure.set(k, frozen_value)
        frozen_func.closure_state = frozen_closure

    return frozen_func


def _rehydrate_user_function(user_func: Any, agent: "Agent") -> Any:
    """Rehydrate a UserFunction by restoring agent reference."""
    from .functions import UserFunction

    if isinstance(user_func, UserFunction):
        user_func.agent = agent

    return user_func


def _freeze_tic_class(tic_class: Any) -> Any:
    """Freeze a TicClass by freezing all its UserFunction methods."""
    from .objects import TicClass

    if not isinstance(tic_class, TicClass):
        return tic_class

    # Create a copy to avoid modifying the original
    frozen_class = copy.copy(tic_class)
    methods_copy = {}

    for method_name, method in frozen_class.methods.items():
        # Freeze each method
        methods_copy[method_name] = ObjectFreezer._freeze_recursive(method)

    frozen_class.methods = methods_copy
    return frozen_class


def _rehydrate_tic_class(tic_class: Any, agent: "Agent") -> Any:
    """Rehydrate a TicClass by rehydrating all its UserFunction methods."""
    from .objects import TicClass

    if isinstance(tic_class, TicClass):
        for method in tic_class.methods.values():
            ObjectFreezer.rehydrate(method, agent)

    return tic_class


def _freeze_tic_instance(tic_instance: Any) -> Any:
    """Freeze a TicInstance by freezing its class and recursively freezing attributes."""
    from .objects import TicInstance

    if not isinstance(tic_instance, TicInstance):
        return tic_instance

    # Create a copy to avoid modifying the original
    frozen_instance = copy.copy(tic_instance)

    # Freeze the class
    frozen_instance.cls = ObjectFreezer._freeze_recursive(frozen_instance.cls)

    # Recursively freeze attributes that might contain unpickleable objects
    frozen_attributes = {}
    for attr_name, attr_value in frozen_instance.attributes.items():
        frozen_attributes[attr_name] = ObjectFreezer._freeze_recursive(attr_value)

    frozen_instance.attributes = frozen_attributes

    return frozen_instance


def _rehydrate_tic_instance(tic_instance: Any, agent: "Agent") -> Any:
    """Rehydrate a TicInstance by rehydrating its class and recursively rehydrating attributes."""
    from .objects import TicInstance

    if isinstance(tic_instance, TicInstance):
        # Rehydrate the class
        ObjectFreezer.rehydrate(tic_instance.cls, agent)

        # Recursively rehydrate attributes
        for attr_name, attr_value in tic_instance.attributes.items():
            rehydrated_value = ObjectFreezer.rehydrate(attr_value, agent)
            tic_instance.attributes[attr_name] = rehydrated_value

    return tic_instance


def _freeze_tic_module(tic_module: Any) -> Any:
    """Freeze a TicModule by converting it to a TicModuleStub."""
    from .objects import TicModule, TicModuleStub

    if isinstance(tic_module, TicModule):
        return TicModuleStub(name=tic_module.__name__)

    return tic_module


def _rehydrate_tic_module_stub(tic_module_stub: Any, agent: "Agent") -> Any:
    """Rehydrate a TicModuleStub back to a TicModule using the agent's registry."""
    from .objects import TicModule, TicModuleStub

    if isinstance(tic_module_stub, TicModuleStub):
        # Try to rebuild the module from the agent's registry using the same logic as _create_tic_module
        reg_module = agent.importable_modules.get(tic_module_stub.name)
        if reg_module:
            # Create a sandboxed module object and populate it properly
            tic_module = TicModule(name=tic_module_stub.name)
            for fn_name in reg_module.fns.keys():
                setattr(tic_module, fn_name, getattr(reg_module.module, fn_name))
            for const_name in reg_module.consts.keys():
                setattr(tic_module, const_name, getattr(reg_module.module, const_name))
            for cls_name, reg_class in reg_module.classes.items():
                setattr(tic_module, cls_name, reg_class.cls)
            return tic_module
        # If not found, return a basic TicModule
        return TicModule(tic_module_stub.name)

    return tic_module_stub


def _freeze_tic_object(tic_object: Any) -> Any:
    """Freeze a TicObject by recursively freezing its attributes."""
    from .objects import TicObject

    if not isinstance(tic_object, TicObject):
        return tic_object

    # Create a copy to avoid modifying the original
    frozen_object = copy.copy(tic_object)

    # Recursively freeze attributes that might contain unpickleable objects
    frozen_attributes = {}
    for attr_name, attr_value in frozen_object.attributes.items():
        frozen_attributes[attr_name] = ObjectFreezer._freeze_recursive(attr_value)

    frozen_object.attributes = frozen_attributes

    return frozen_object


def _rehydrate_tic_object(tic_object: Any, agent: "Agent") -> Any:
    """Rehydrate a TicObject by recursively rehydrating its attributes."""
    from .objects import TicObject

    if isinstance(tic_object, TicObject):
        # Recursively rehydrate attributes
        for attr_name, attr_value in tic_object.attributes.items():
            rehydrated_value = ObjectFreezer.rehydrate(attr_value, agent)
            tic_object.attributes[attr_name] = rehydrated_value

    return tic_object


def _freeze_tic_dataclass(tic_dataclass: Any) -> Any:
    """Freeze a TicDataClass - no special handling needed, just return as-is."""
    return tic_dataclass


def _rehydrate_tic_dataclass(tic_dataclass: Any, agent: "Agent") -> Any:
    """Rehydrate a TicDataClass - no special handling needed."""
    return tic_dataclass


def register_eval_handlers() -> None:
    """Register all eval object freeze/rehydrate handlers."""
    ObjectFreezer.register_handler(
        "UserFunction", _freeze_user_function, _rehydrate_user_function
    )

    ObjectFreezer.register_handler("TicClass", _freeze_tic_class, _rehydrate_tic_class)

    ObjectFreezer.register_handler(
        "TicInstance", _freeze_tic_instance, _rehydrate_tic_instance
    )

    ObjectFreezer.register_handler(
        "TicModule", _freeze_tic_module, _rehydrate_tic_module_stub
    )

    ObjectFreezer.register_handler(
        "TicModuleStub",
        lambda x: x,  # TicModuleStub doesn't need freezing
        _rehydrate_tic_module_stub,
    )

    ObjectFreezer.register_handler(
        "TicDataClass", _freeze_tic_dataclass, _rehydrate_tic_dataclass
    )

    ObjectFreezer.register_handler(
        "TicObject", _freeze_tic_object, _rehydrate_tic_object
    )


# Auto-register handlers when this module is imported
register_eval_handlers()
