"""
User-facing errors that can be caught within the tic evaluation environment.
"""

import ast
from typing import Optional


class AgexError(Exception):
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


class AgexValueError(AgexError):
    """Raised when a function receives an argument of the right type but an inappropriate value."""

    pass


class AgexTypeError(AgexError):
    """Raised when an operation or function is applied to an object of inappropriate type."""

    pass


class AgexKeyError(AgexError):
    """Raised when a mapping (dictionary) key is not found."""

    pass


class AgexIndexError(AgexError):
    """Raised when a sequence subscript is out of range."""

    pass


class AgexAttributeError(AgexError):
    """Raised when an attribute reference or assignment fails."""

    pass


class AgexNameError(AgexError):
    """Raised when a local or global name is not found."""

    pass


class AgexZeroDivisionError(AgexError):
    """Raised when the second argument of a division or modulo operation is zero."""

    pass


class AgexArithmeticError(AgexError):
    """Base class for arithmetic errors (overflow, underflow, etc.)."""

    pass


class AgexOverflowError(AgexArithmeticError):
    """Raised when the result of an arithmetic operation is too large to be represented."""

    pass
