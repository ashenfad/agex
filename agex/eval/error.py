class EvalError(Exception):
    """Custom exception for evaluation errors."""

    def __init__(self, message: str, cause: Exception | None = None):
        self.message = message
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        return self.message
