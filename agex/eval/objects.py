"""
Internal representation of objects used by the bridge and render layers.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Union

from .user_errors import AgexAttributeError, AgexError


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
            method_spec = self.reg_object.methods[name]
            return BoundInstanceMethod(
                reg_object=self.reg_object,
                host_registry=self.host_registry,
                method_name=name,
                host_fs_access=getattr(method_spec, "host_fs_access", False),
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
    host_fs_access: bool = False
    network_access: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Look up the live object and call the real method."""
        live_instance = self.host_registry[self.reg_object.name]
        method = getattr(live_instance, self.method_name)
        try:
            return method(*args, **kwargs)
        except Exception as e:  # Map to agent-catchable errors
            # Pass through already-wrapped agent errors
            if isinstance(e, AgexError):
                raise
            # Specific mappings take precedence
            for src_exc, target_exc in self.reg_object.exception_mappings.items():
                if isinstance(e, src_exc):
                    raise target_exc(str(e)) from e
            # Fallback: wrap into generic AgexError with original type name
            raise AgexError(f"{type(e).__name__}: {e}") from e


@dataclass
class AgexModule:
    """A serializable module reference for use within the agex bridge."""

    name: str
    agent_fingerprint: str = (
        ""  # Parent agent who registered this module (for security inheritance)
    )
    submodules: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"<agexmodule '{self.name}'>"

    def getattr(self, name: str) -> Any:
        if name in self.submodules:
            return self.submodules[name]
        raise AgexAttributeError(f"module '{self.name}' has no attribute '{name}'")

    def setattr(self, name: str, value: Any):
        # Allow attaching submodules
        self.submodules[name] = value


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
