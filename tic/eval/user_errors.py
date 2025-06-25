"""
User-facing errors that can be caught within the tic evaluation environment.
"""

import ast
from typing import Optional


class TicError(Exception):
    """Base class for all user-catchable errors in tic."""

    def __init__(self, message: str, node: Optional[ast.AST] = None):
        super().__init__(message)
        self.message = message
        self.node = node

    def __str__(self):
        if (
            self.node
            and hasattr(self.node, "lineno")
            and hasattr(self.node, "col_offset")
        ):
            return f"Error at line {self.node.lineno}, col {self.node.col_offset}: {self.message}"  # type: ignore
        return self.message


class TicValueError(TicError):
    """Raised when a function receives an argument of the right type but an inappropriate value."""

    pass


class TicTypeError(TicError):
    """Raised when an operation or function is applied to an object of inappropriate type."""

    pass


class TicKeyError(TicError):
    """Raised when a mapping (dictionary) key is not found."""

    pass


class TicIndexError(TicError):
    """Raised when a sequence subscript is out of range."""

    pass


class TicAttributeError(TicError):
    """Raised when an attribute reference or assignment fails."""

    pass


class TicNameError(TicError):
    """Raised when a local or global name is not found."""

    pass
