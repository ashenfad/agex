"""
Internal representation of user-defined objects (dataclasses).
"""

from dataclasses import dataclass
from typing import Any

from .error import EvalError


@dataclass
class TicDataClass:
    """Represents a dataclass definition. It's a callable factory for TicObjects."""

    name: str
    fields: dict[str, Any]

    def __call__(self, *args: Any, **kwargs: Any) -> "TicObject":
        """Creates an instance of this dataclass."""
        if len(args) > len(self.fields):
            raise EvalError(
                f"{self.name}() takes {len(self.fields)} positional arguments but {len(args)} were given",
                node=None,
            )

        bound_args = {}
        # Simple argument binding: first by position, then by keyword.
        for i, field_name in enumerate(self.fields):
            if i < len(args):
                if field_name in kwargs:
                    raise EvalError(
                        f"{self.name}() got multiple values for argument '{field_name}'",
                        node=None,
                    )
                bound_args[field_name] = args[i]
            elif field_name in kwargs:
                bound_args[field_name] = kwargs.pop(field_name)
            else:
                raise EvalError(
                    f"{self.name}() missing required positional argument: '{field_name}'",
                    node=None,
                )

        if kwargs:
            unexpected = next(iter(kwargs))
            raise EvalError(
                f"{self.name}() got an unexpected keyword argument '{unexpected}'",
                node=None,
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
            raise EvalError(
                f"'{self.cls.name}' object has no attribute '{name}'", node=None
            )
        return self.attributes[name]

    def setattr(self, name: str, value: Any):
        if name not in self.cls.fields:
            raise EvalError(
                f"'{self.cls.name}' object has no attribute '{name}' (cannot add new attributes)",
                node=None,
            )
        self.attributes[name] = value


class TicModule:
    """A sandboxed module object for use within the Tic evaluator."""

    def __init__(self, name: str):
        self.__name__ = name

    def __repr__(self):
        return f"<ticmodule '{self.__name__}'>"


class PrintTuple(tuple):
    """A wrapper to distinguish tuples created by print() for special rendering."""

    pass
