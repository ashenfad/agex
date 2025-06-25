"""
Internal representation of user-defined objects (dataclasses).
"""

from dataclasses import dataclass, field
from typing import Any

from .user_errors import TicAttributeError, TicTypeError


@dataclass
class TicDataClass:
    """Represents a dataclass definition. It's a callable factory for TicObjects."""

    name: str
    fields: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> "TicObject":
        """Creates an instance of this dataclass."""
        if len(args) > len(self.fields):
            raise TicTypeError(
                f"{self.name}() takes {len(self.fields)} positional arguments but {len(args)} were given"
            )

        bound_args = {}
        # Simple argument binding: first by position, then by keyword.
        for i, field_name in enumerate(self.fields):
            if i < len(args):
                if field_name in kwargs:
                    raise TicTypeError(
                        f"{self.name}() got multiple values for argument '{field_name}'"
                    )
                bound_args[field_name] = args[i]
            elif field_name in kwargs:
                bound_args[field_name] = kwargs.pop(field_name)
            else:
                raise TicTypeError(
                    f"{self.name}() missing required positional argument: '{field_name}'"
                )

        if kwargs:
            unexpected = next(iter(kwargs))
            raise TicTypeError(
                f"{self.name}() got an unexpected keyword argument '{unexpected}'"
            )

        return TicObject(cls=self, attributes=bound_args)


@dataclass
class TicObject:
    """Represents an instance of a TicDataClass."""

    cls: TicDataClass
    attributes: dict[str, Any]

    def __repr__(self) -> str:
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.attributes.items())
        return f"{self.cls.name}({attrs})"

    def getattr(self, name: str) -> Any:
        if name not in self.attributes:
            raise TicAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        return self.attributes[name]

    def setattr(self, name: str, value: Any):
        if name not in self.cls.fields:
            raise TicAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}' (cannot add new attributes)"
            )
        self.attributes[name] = value

    def delattr(self, name: str):
        if name not in self.attributes:
            raise TicAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        del self.attributes[name]


class TicClass:
    """Represents a user-defined class created with the 'class' keyword."""

    def __init__(self, name: str, methods: dict[str, Any]):
        self.name = name
        self.methods = methods

    def __repr__(self):
        return f"<class '{self.name}'>"

    def __setstate__(self, state):
        """Custom unpickle behavior - restore all fields."""
        self.__dict__.update(state)

    def __call__(self, *args: Any, **kwargs: Any) -> "TicInstance":
        """Create an instance of the class."""
        instance = TicInstance(cls=self)

        # Look for an __init__ method and call it if it exists.
        if "__init__" in self.methods:
            init_method = self.methods["__init__"]
            bound_init = TicMethod(instance=instance, function=init_method)
            bound_init(*args, **kwargs)  # Call __init__

        return instance


@dataclass
class TicInstance:
    """Represents an instance of a user-defined TicClass."""

    cls: TicClass
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
            return TicMethod(instance=self, function=function)

        raise TicAttributeError(f"'{self.cls.name}' object has no attribute '{name}'")

    def setattr(self, name: str, value: Any):
        """Set an attribute on the instance."""
        self.attributes[name] = value

    def delattr(self, name: str):
        """Delete an attribute from the instance."""
        if name not in self.attributes:
            raise TicAttributeError(
                f"'{self.cls.name}' object has no attribute '{name}'"
            )
        del self.attributes[name]


@dataclass
class TicMethod:
    """A method bound to a TicInstance. It's a callable wrapper."""

    instance: TicInstance
    function: Any  # This will be a tic.eval.functions.UserFunction

    def __call__(self, *args, **kwargs):
        """Call the underlying function with the instance as the first argument."""
        # This allows TicMethod to wrap any callable, not just UserFunction.
        return self.function(self.instance, *args, **kwargs)


@dataclass
class TicModule:
    """A sandboxed, serializable module object for use within the Tic evaluator."""

    name: str

    def __repr__(self):
        return f"<ticmodule '{self.name}'>"


class PrintTuple(tuple):
    """A wrapper to distinguish tuples created by print() for special rendering."""

    pass
