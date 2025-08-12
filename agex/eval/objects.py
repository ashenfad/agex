"""
Internal representation of user-defined objects (dataclasses).
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Union

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

    def setattr(self, name: str, value: Any):
        """Set an attribute on the live host object."""
        # Check if this attribute is registered as a property
        if name not in self.reg_object.properties:
            raise AgexAttributeError(
                f"'{self.reg_object.name}' object has no registered property '{name}'"
            )

        live_instance = self.host_registry[self.reg_object.name]
        setattr(live_instance, name, value)

    def delattr(self, name: str):
        """Delete an attribute from the live host object."""
        # Check if this attribute is registered as a property
        if name not in self.reg_object.properties:
            raise AgexAttributeError(
                f"'{self.reg_object.name}' object has no registered property '{name}'"
            )

        live_instance = self.host_registry[self.reg_object.name]
        delattr(live_instance, name)

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
        try:
            return method(*args, **kwargs)
        except Exception as e:  # Map to agent-catchable errors
            from agex.agent.datatypes import _AgentExit

            from .user_errors import AgexError

            # Pass through agent control and already agent errors
            if isinstance(e, (_AgentExit, AgexError)):
                raise
            # Specific mappings take precedence
            for src_exc, target_exc in self.reg_object.exception_mappings.items():
                if isinstance(e, src_exc):
                    raise target_exc(str(e)) from e
            # Fallback: wrap into generic AgexError with original type name
            raise AgexError(f"{type(e).__name__}: {e}") from e

    # New unified execution hook used by the evaluator
    def execute(self, args: list[Any], kwargs: dict[str, Any]) -> Any:
        live_instance = self.host_registry[self.reg_object.name]
        method = getattr(live_instance, self.method_name)
        try:
            return method(*args, **kwargs)
        except Exception as e:  # Map to agent-catchable errors
            from agex.agent.datatypes import _AgentExit

            from .user_errors import AgexError

            if isinstance(e, (_AgentExit, AgexError)):
                raise
            for src_exc, target_exc in self.reg_object.exception_mappings.items():
                if isinstance(e, src_exc):
                    raise target_exc(str(e)) from e
            raise AgexError(f"{type(e).__name__}: {e}") from e


@dataclass
class AgexModule:
    """A sandboxed, serializable module object for use within the Agex evaluator."""

    name: str
    agent_fingerprint: str = (
        ""  # Parent agent who registered this module (for security inheritance)
    )

    def __repr__(self):
        return f"<agexmodule '{self.name}'>"


class PrintAction(tuple):
    """Represents the un-rendered content of a print() call."""

    pass


@dataclass
class ImageAction:
    """Represents an un-rendered image from a view_image() call."""

    image: Any
    detail: Literal["low", "high"] = "high"

    def _repr_html_(self) -> str:
        """Rich HTML representation for notebook display."""
        # First, try the object's native _repr_html_ method (e.g., plotly figures)
        if hasattr(self.image, "_repr_html_"):
            try:
                return self.image._repr_html_()
            except Exception:
                pass  # Fall through to image serialization

        # For other image types, convert to base64 and display as HTML image
        try:
            # Import here to avoid circular dependency
            from agex.render.stream import _serialize_image_to_base64

            base64_image = _serialize_image_to_base64(self.image)
            if base64_image:
                return f'<img src="data:image/png;base64,{base64_image}" style="max-width: 100%; height: auto;" />'
        except Exception:
            pass  # Fall through to text fallback

        # Fallback to text representation
        import html

        type_name = type(self.image).__name__
        escaped_text = html.escape(f"<{type_name} image - display failed>")
        return f'<pre style="background: #f6f8fa; padding: 8px; border-radius: 6px; margin: 0; color: #24292e; font-family: monospace;">{escaped_text}</pre>'


ContentPart = Union[PrintAction, ImageAction]
