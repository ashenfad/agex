class EvalError(Exception):
    """Custom exception for evaluation errors."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message


class ScopeRequired(Exception):
    """Raised when agent code uses a registered capability whose ``scope``
    has not been granted in the current session.

    This is an *ordinary* exception, not a control exception: agex's loop
    feeds it back to the agent as a normal error observation, and the agent
    reacts by explicitly calling ``task_request_permission(scope=...)`` (or
    pivoting). It is never auto-converted into a request.
    """

    def __init__(self, message: str, *, scope: str, name: str | None = None):
        self.message = message
        self.scope = scope
        self.name = name
        super().__init__(self.message)

    def __str__(self):
        return self.message
