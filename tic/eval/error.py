import ast


class EvalError(Exception):
    """Custom exception for evaluation errors."""

    def __init__(
        self, message: str, node: ast.AST | None, cause: Exception | None = None
    ):
        self.message = message
        self.node = node
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        if (
            self.node
            and hasattr(self.node, "lineno")
            and hasattr(self.node, "col_offset")
        ):
            return f"Error at line {self.node.lineno}, col {self.node.col_offset}: {self.message}"  # type: ignore
        return self.message
