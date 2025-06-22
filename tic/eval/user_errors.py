"""
User-facing errors that can be caught within the tic evaluation environment.
"""


class TicError(Exception):
    """Base class for all user-catchable errors in tic."""

    pass


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
