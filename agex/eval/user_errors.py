"""
User-facing errors that can be caught within the agex evaluation environment.
"""


class AgexError(Exception):
    """Base class for all user-catchable errors in agex."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class AgexValueError(AgexError):
    """Raised when a function receives an argument of the right type but an inappropriate value."""

    pass


class AgexTypeError(AgexError):
    """Raised when an operation or function is applied to an object of inappropriate type."""

    pass


class AgexAttributeError(AgexError):
    """Raised when an attribute reference or assignment fails."""

    pass
