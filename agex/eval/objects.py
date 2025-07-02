"""
Internal representation of user-defined objects (dataclasses).
"""

from dataclasses import dataclass, field
from typing import Any

from .user_errors import AgexAttributeError, AgexTypeError


@dataclass
class AgexDataClass:
    """Represents a dataclass definition. It's a callable factory for AgexObjects."""

    name: str
    fields: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> "AgexObject":
        """Creates an instance of this dataclass."""
        if len(args) > len(self.fields):
            raise AgexTypeError(
                f"{self.name}() takes {len(self.fields)} positional arguments but {len(args)} were given"
            )

        bound_args = {}
        # Simple argument binding: first by position, then by keyword.
        for i, field_name in enumerate(self.fields):
            if i < len(args):
                if field_name in kwargs:
                    raise AgexTypeError(
                        f"{self.name}() got multiple values for argument '{field_name}'"
                    )
                bound_args[field_name] = args[i]
            elif field_name in kwargs:
                bound_args[field_name] = kwargs.pop(field_name)
            else:
                raise AgexTypeError(
                    f"{self.name}() missing required positional argument: '{field_name}'"
                )

        if kwargs:
            unexpected = next(iter(kwargs))
            raise AgexTypeError(
                f"{self.name}() got an unexpected keyword argument '{unexpected}'"
            )

        return AgexObject(cls=self, attributes=bound_args)


@dataclass
class AgexObject:
    """Represents an instance of a AgexDataClass."""

    cls: AgexDataClass
    attributes: dict[str, Any]

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.cls.name}({attrs})"

    def getattr(self, name: str) -> Any:
        if name not in self.attributes:
            raise AgexAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        return self.attributes[name]

    def setattr(self, name: str, value: Any):
        if name not in self.cls.fields:
            raise AgexAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}' (cannot add new attributes)"
            )
        self.attributes[name] = value

    def delattr(self, name: str):
        if name not in self.attributes:
            raise AgexAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        del self.attributes[name]


class AgexClass:
    """Represents a user-defined class created with the 'class' keyword."""

    def __init__(self, name: str, methods: dict[str, Any]):
        self.name = name
        self.methods = methods

    def __repr__(self):
        return f"<class '{self.name}'>"

    def __setstate__(self, state):
        """Custom unpickle behavior - restore all fields."""
        self.__dict__.update(state)

    def __call__(self, *args: Any, **kwargs: Any) -> "AgexInstance":
        """Create an instance of the class."""
        instance = AgexInstance(cls=self)

        # Look for an __init__ method and call it if it exists.
        if "__init__" in self.methods:
            init_method = self.methods["__init__"]
            bound_init = AgexMethod(instance=instance, function=init_method)
            bound_init(*args, **kwargs)  # Call __init__

        return instance


@dataclass
class AgexInstance:
    """Represents an instance of a user-defined AgexClass."""

    cls: AgexClass
    attributes: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<{self.cls.name} object>"

    def getattr(self, name: str) -> Any:
        """Get an attribute from the instance, or a method from the class."""
        # Instance attributes take precedence
        if name in self.attributes:
            return self.attributes[name]

        # Then, look for a method on the class
        if name in self.cls.methods:
            function = self.cls.methods[name]
            return AgexMethod(instance=self, function=function)

        raise AgexAttributeError(f"'{self.cls.name}' object has no attribute '{name}'")

    def setattr(self, name: str, value: Any):
        """Set an attribute on the instance."""
        self.attributes[name] = value

    def delattr(self, name: str):
        """Delete an attribute from the instance."""
        if name not in self.attributes:
            raise AgexAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        del self.attributes[name]


@dataclass
class AgexMethod:
    """A method bound to a AgexInstance. It's a callable wrapper."""

    instance: AgexInstance
    function: Any  # This will be a tic.eval.functions.UserFunction

    def __call__(self, *args, **kwargs):
        """Call the underlying function with the instance as the first argument."""
        # This allows AgexMethod to wrap any callable, not just UserFunction.
        return self.function(self.instance, *args, **kwargs)


@dataclass
class BoundInstanceObject:
    """A proxy for a live host object, exposing its methods and properties."""

    reg_object: Any  # RegisteredObject
    host_registry: dict[str, Any]

    def __repr__(self) -> str:
        return f"<live_object '{self.reg_object.name}'>"

    def getattr(self, name: str) -> Any:
        """Get a method or property from the live host object."""
        if name in self.reg_object.methods:
            return BoundInstanceMethod(
                reg_object=self.reg_object,
                host_registry=self.host_registry,
                method_name=name,
            )
        if name in self.reg_object.properties:
            live_instance = self.host_registry[self.reg_object.name]
            return getattr(live_instance, name)

        raise AgexAttributeError(
            f"'{self.reg_object.name}' object has no attribute '{name}'"
        )

    def __enter__(self):
        """Context manager entry - delegate to the live object if it supports it."""
        live_instance = self.host_registry[self.reg_object.name]
        if hasattr(live_instance, "__enter__"):
            # Call the live object's __enter__ method
            enter_result = live_instance.__enter__()
            # If the live object returns itself (common pattern), return our proxy instead
            # so that method access continues to go through our controlled interface
            if enter_result is live_instance:
                return self
            else:
                # If the live object returns something else (like a value), return that
                return enter_result
        else:
            # If the live object doesn't support context manager protocol,
            # we can still provide basic support by returning the proxy object
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - delegate to the live object if it supports it."""
        live_instance = self.host_registry[self.reg_object.name]
        if hasattr(live_instance, "__exit__"):
            return live_instance.__exit__(exc_type, exc_val, exc_tb)
        else:
            # If the live object doesn't have __exit__, we don't suppress exceptions
            return False


@dataclass
class BoundInstanceMethod:
    """A callable proxy for a method on a live host object."""

    reg_object: Any  # RegisteredObject
    host_registry: dict[str, Any]
    method_name: str

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Look up the live object and call the real method."""
        live_instance = self.host_registry[self.reg_object.name]
        method = getattr(live_instance, self.method_name)
        return method(*args, **kwargs)


@dataclass
class AgexModule:
    """A sandboxed, serializable module object for use within the Agex evaluator."""

    name: str
    agent_fingerprint: str = (
        ""  # Parent agent who registered this module (for security inheritance)
    )

    def __repr__(self):
        return f"<agexmodule '{self.name}'>"


class PrintTuple(tuple):
    """A wrapper to distinguish tuples created by print() for special rendering."""

    pass
