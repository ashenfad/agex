import ast
import inspect
from typing import Callable


def is_function_body_empty(func: Callable) -> bool:
    """
    Check if a function body contains only pass statements, docstrings, and comments.

    Returns True if the function body is effectively empty (suitable for @agent.task).
    """
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
                func_def = node
                break

        if not func_def:
            return False

        # Check the function body
        for stmt in func_def.body:
            if isinstance(stmt, ast.Pass):
                continue
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                # Docstring (string literal as expression)
                continue
            else:
                # Found a non-trivial statement
                return False

        return True
    except (OSError, TypeError):
        # Can't get source (built-in, dynamically created, etc.)
        # Be conservative and assume it's not empty
        return False
