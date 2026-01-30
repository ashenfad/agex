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


def compute_agex_mro(agex_cls: "AgexClass") -> list[Union["AgexClass", type]]:
    """Compute MRO using Python's C3 linearization via stub classes.

    For mixed hierarchies (AgexClass inheriting from host classes and other AgexClasses),
    we create temporary Python class stubs that mirror the hierarchy, let Python compute
    the MRO, then map back to our AgexClass objects.
    """
    stub_cache: dict["AgexClass", type] = {}

    def get_stub(cls: Union["AgexClass", type]) -> type:
        if isinstance(cls, type):
            # Already a real Python class
            return cls
        if cls in stub_cache:
            return stub_cache[cls]
        # Create stub with correct bases
        stub_bases = tuple(get_stub(b) for b in cls.bases) or (object,)
        stub = type(cls.name, stub_bases, {})
        stub_cache[cls] = stub
        return stub

    # Get MRO from the stub
    root_stub = get_stub(agex_cls)
    python_mro = root_stub.__mro__

    # Map back to AgexClass where applicable
    reverse_map = {v: k for k, v in stub_cache.items()}
    return [reverse_map.get(c, c) for c in python_mro]


class AgexClass:
    """Represents a user-defined class created with the 'class' keyword."""

    def __init__(
        self,
        name: str,
        methods: dict[str, Any],
        bases: Union[list[Union[type, "AgexClass"]], None] = None,
        local_instance_attrs: Union[set[str], None] = None,
    ):
        self.name = name
        self.methods = methods
        self.bases = bases or []
        self.local_instance_attrs = local_instance_attrs or set()
        self.mro: list[Union[type, "AgexClass"]] = []  # Computed after init

    def __repr__(self):
        return f"<class '{self.name}'>"

    def __setstate__(self, state):
        """Custom unpickle behavior - restore all fields."""
        self.__dict__.update(state)

    def __call__(self, *args: Any, **kwargs: Any) -> "AgexInstance":
        """Create an instance of the class."""
        instance = AgexInstance(cls=self)

        # Create host proxy if we have a host base
        host_base = self._first_host_base()
        if host_base is not None:
            instance._host_proxy = object.__new__(host_base)

        # Look for an __init__ method and call it if it exists.
        if "__init__" in self.methods:
            init_method = self.methods["__init__"]
            bound_init = AgexMethod(instance=instance, function=init_method)
            bound_init(*args, **kwargs)  # Call __init__

        return instance

    def _first_host_base(self) -> type | None:
        """Find the first host (non-AgexClass) base in the MRO."""
        for cls in self.mro:
            if isinstance(cls, type) and cls is not object:
                return cls
        return None


@dataclass
class AgexInstance:
    """Represents an instance of a user-defined AgexClass."""

    cls: AgexClass
    attributes: dict[str, Any] = field(default_factory=dict)
    _host_proxy: Any | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"<{self.cls.name} object>"

    def getattr(self, name: str, agent: Any = None) -> Any:
        """Get an attribute from the instance with MRO-aware policy checks.

        Args:
            name: Attribute name
            agent: Optional agent for policy checking when accessing inherited host attrs

        Returns:
            The attribute value or bound method
        """
        # 1. Check if this is an instance attribute
        if name in self.attributes:
            # Verify it's allowed by walking MRO to find which class declared it
            for cls in self.cls.mro:
                if isinstance(cls, AgexClass):
                    # For AgexClass: allowed if in local_instance_attrs
                    if name in cls.local_instance_attrs:
                        return self.attributes[name]
                elif agent is not None:
                    # For host class: allowed if policy approves
                    if agent._policy.resolve_class_member(cls, name) is not None:
                        return self.attributes[name]

            # If not declared by any class in MRO, it was set dynamically on this instance
            # For classes with no bases (or only object), allow it
            # For classes with bases, only allow if this is the immediate class
            if len(self.cls.mro) <= 2:  # Just [self, object]
                return self.attributes[name]

            # Has bases - check if this was set locally (not inherited)
            # We can't distinguish, so for safety: allow it (dynamic attributes are a Python feature)
            # But log that this could be a security issue if we want stricter control
            return self.attributes[name]

        # 2. Local methods
        if name in self.cls.methods:
            function = self.cls.methods[name]
            return AgexMethod(instance=self, function=function)

        # 3. Walk MRO for inherited methods
        for cls in self.cls.mro[1:]:  # Skip self (first in MRO)
            if isinstance(cls, AgexClass):
                # Sandbox base - check its methods
                if name in cls.methods:
                    return AgexMethod(instance=self, function=cls.methods[name])
            elif agent is not None:
                # Host base - check policy for methods/descriptors on host proxy
                if agent._policy.resolve_class_member(cls, name) is not None:
                    if self._host_proxy is not None:
                        try:
                            value = getattr(self._host_proxy, name)
                            return value
                        except AttributeError:
                            pass

        raise AgexAttributeError(f"'{self.cls.name}' object has no attribute '{name}'")

    def setattr(self, name: str, value: Any, agent: Any = None):
        """Set an attribute on the instance.

        Args:
            name: Attribute name
            value: Attribute value
            agent: Optional agent for policy checking when syncing to host proxy
        """
        # Check if this attribute name exists in the parent classes
        # If it does and isn't whitelisted, block the write
        if agent is not None and len(self.cls.mro) > 2:  # Has inheritance beyond object
            for cls in self.cls.mro[1:]:  # Skip self
                if isinstance(cls, AgexClass):
                    # For AgexClass parents: allow if in local_instance_attrs
                    if name in cls.local_instance_attrs:
                        break  # Found and allowed
                    # Check methods - can't set attributes shadowing methods
                    if name in cls.methods:
                        raise AgexAttributeError(
                            f"Cannot set attribute '{name}' - it's a method on parent class '{cls.name}'"
                        )
                else:
                    # For host parents: check policy first
                    member_result = agent._policy.resolve_class_member(cls, name)
                    if member_result is not None:
                        # Attribute exists and is whitelisted - allow
                        break

                    # Policy returned None - check if attribute exists on the class
                    # If it exists but isn't whitelisted, block it
                    if hasattr(cls, name):
                        # Exists on parent but not whitelisted - block it
                        raise AgexAttributeError(
                            f"Cannot set attribute '{name}' - it exists on parent class "
                            f"'{cls.__name__}' but is not whitelisted"
                        )

        # Passed policy checks - allow the write
        self.attributes[name] = value

        # Sync to host proxy if applicable
        if self._host_proxy is not None and agent is not None:
            for cls in self.cls.mro:
                if isinstance(cls, type):
                    if agent._policy.resolve_class_member(cls, name) is not None:
                        try:
                            setattr(self._host_proxy, name, value)
                        except (AttributeError, TypeError):
                            # Some attrs may not be settable
                            pass
                        break

    def delattr(self, name: str):
        """Delete an attribute from the instance."""
        if name not in self.attributes:
            raise AgexAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        del self.attributes[name]


@dataclass
class SuperProxy:
    """Proxy returned by super() for navigating the MRO."""

    instance: AgexInstance
    remaining_mro: list[Union[AgexClass, type]]
    agent: Any

    def getattr(self, name: str) -> Any:
        """Get an attribute from the remaining MRO."""
        for cls in self.remaining_mro:
            if isinstance(cls, AgexClass):
                # Sandbox class - check methods
                if name in cls.methods:
                    return AgexMethod(
                        instance=self.instance, function=cls.methods[name]
                    )
            else:
                # Host class - check policy
                member_result = self.agent._policy.resolve_class_member(cls, name)
                if member_result is not None:
                    # Policy allows this member - now get it
                    if hasattr(cls, name):
                        member = getattr(cls, name)
                        # If it's a method/function, we need to bind it to host_proxy
                        if callable(member) and self.instance._host_proxy is not None:
                            # Manually bind: create a bound method
                            import types

                            if isinstance(
                                member, (types.FunctionType, types.MethodType)
                            ):
                                # It's an unbound method or descriptor, bind it
                                bound = member.__get__(
                                    self.instance._host_proxy,
                                    type(self.instance._host_proxy),
                                )
                                return bound
                        return member

        raise AgexAttributeError(f"'super' object has no attribute '{name}'")


@dataclass
class AgexMethod:
    """A method bound to a AgexInstance. It's a callable wrapper."""

    instance: AgexInstance
    function: Any  # This will be a tic.eval.functions.UserFunction
    defining_class: Union[AgexClass, None] = field(default=None, init=False)

    def __post_init__(self):
        """Find which class in the MRO defines this method."""
        for cls in self.instance.cls.mro:
            if isinstance(cls, AgexClass) and self.function in cls.methods.values():
                self.defining_class = cls
                break
        if self.defining_class is None:
            self.defining_class = self.instance.cls

    def __call__(self, *args, **kwargs):
        """Call the underlying function with the instance as the first argument."""
        # Store method context for super() support
        # UserFunction.execute will pick this up and set it on the evaluator
        self.instance._method_context = (self.defining_class, self.instance)
        try:
            result = self.function(self.instance, *args, **kwargs)
        finally:
            self.instance._method_context = None
        return result


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
    submodules: dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"<agexmodule '{self.name}'>"

    def getattr(self, name: str) -> Any:
        if name in self.submodules:
            return self.submodules[name]
        # JIT resolution will handle other attributes via Resolver
        raise AgexAttributeError(f"module '{self.name}' has no attribute '{name}'")

    def setattr(self, name: str, value: Any):
        # Allow attaching submodules
        self.submodules[name] = value


@dataclass
class AgexVFSModule:
    """A module object backed by a Namespaced VFS state."""

    name: str
    state: Any  # Namespaced state (runtime only, None when detached)
    agent_fingerprint: str | None = None
    session: str = "default"

    def __repr__(self):
        return f"<module '{self.name}' (VFS)>"

    def __getstate__(self):
        """Custom pickling to avoid serializing the live state object."""
        return {
            "name": self.name,
            "agent_fingerprint": self.agent_fingerprint,
            "session": self.session,
        }

    def __setstate__(self, state):
        """Restore metadata and start in detached state."""
        self.name = state["name"]
        self.agent_fingerprint = state.get("agent_fingerprint")
        self.session = state.get("session", "default")
        self.state = None  # Detached

    def _ensure_attached(self):
        """Re-attach to the agent's state if detached."""
        if self.state is not None:
            return

        if not self.agent_fingerprint:
            raise RuntimeError(
                f"Cannot rehydrate VFS module '{self.name}': missing agent fingerprint."
            )

        from agex.agent.base import resolve_agent
        from agex.state import Namespaced

        # Resolve the agent
        try:
            agent = resolve_agent(self.agent_fingerprint)
        except RuntimeError as e:
            raise RuntimeError(f"Cannot rehydrate VFS module '{self.name}': {e}") from e

        # Get the agent's committed state for the specific session
        # Note: This creates a new Versioned view on the shared store.
        base = agent.state(session=self.session).base_store

        # Reconstruct the namespace hierarchy: modules/<name>
        root_ns = Namespaced(base, "modules")
        self.state = Namespaced(root_ns, self.name)

    def getattr(self, name: str) -> Any:
        self._ensure_attached()
        # Check for members in the namespaced state
        if name in self.state:
            return self.state.get(name)

        raise AgexAttributeError(f"module '{self.name}' has no attribute '{name}'")

    def setattr(self, name: str, value: Any):
        self._ensure_attached()
        self.state.set(name, value)


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
