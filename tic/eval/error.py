import ast


class EvalError(Exception):
    """Custom exception for evaluation errors."""

    def __init__(self, message: str, node: ast.AST, cause: Exception | None = None):
        self.message = message
        self.node = node
        self.cause = cause
        super().__init__(self.message)

    def __str__(self):
        return f"Error at line {self.node.lineno}, col {self.node.col_offset}: {self.message}"
