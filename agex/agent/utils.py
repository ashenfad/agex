import ast
import asyncio
import inspect
import textwrap
from typing import Awaitable, Callable, TypeVar


def is_function_body_empty(func: Callable) -> bool:
    """
    Check if a function body contains only pass statements, docstrings, and comments.

    Returns True if the function body is effectively empty (suitable for @agent.task).
    """
    try:
        source = inspect.getsource(func)

        # Handle indentation issues by dedenting the source
        source = textwrap.dedent(source)

        # If source starts with @, find the function definition
        lines = source.strip().split("\n")
        func_start = 0
        for i, line in enumerate(lines):
            # Check for both synchronous and asynchronous definitions
            if (
                line.strip().startswith("def ") or line.strip().startswith("async def ")
            ) and func.__name__ in line:
                func_start = i
                break

        # Extract just the function definition (not decorators)
        func_source = "\n".join(lines[func_start:])

        tree = ast.parse(func_source)

        # Find the function definition
        func_def = None
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func.__name__
            ):
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
    except (OSError, TypeError, SyntaxError, IndentationError):
        # Can't get source (built-in, dynamically created, etc.) or parse issues
        # Be conservative and assume it's not empty
        return False


def get_instance_attributes_from_init(py_cls: type) -> set[str]:
    """Extract instance attributes assigned in __init__ across the entire MRO.

    Parses the source of each __init__ method and collects assignments to
    self.<attr>. Best-effort: failures to inspect or parse are ignored.
    """
    attributes: set[str] = set()

    for base_cls in py_cls.__mro__:
        if not hasattr(base_cls, "__init__") or base_cls.__init__ is object.__init__:
            continue

        try:
            source = inspect.getsource(base_cls.__init__)
            source = textwrap.dedent(source)
            tree = ast.parse(source)

            class AttributeVisitor(ast.NodeVisitor):
                def visit_Assign(self, node):  # type: ignore[override]
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                        ):
                            attributes.add(target.attr)
                    self.generic_visit(node)

                def visit_AnnAssign(self, node):  # type: ignore[override]
                    # Handle annotated assignments like: self.attr: type = value
                    if (
                        isinstance(node.target, ast.Attribute)
                        and isinstance(node.target.value, ast.Name)
                        and node.target.value.id == "self"
                    ):
                        attributes.add(node.target.attr)
                    self.generic_visit(node)

            AttributeVisitor().visit(tree)
        except Exception:
            # Ignore classes where source isn't available or parse fails
            continue

    return attributes


T = TypeVar("T")
R = TypeVar("R")


def call_sync_or_async(
    handler: Callable[[T], R | Awaitable[R]],
    arg: T,
    loop: asyncio.AbstractEventLoop | None = None,
) -> Awaitable[R] | R | None:
    """
    Call a handler that might be sync or async.

    If the handler is async (coroutine function):
        - If loop is provided (threaded context): Schedule using run_coroutine_threadsafe
        - If loop is None (async context): Return awaitable (must be awaited by caller)

    If the handler is sync:
        - Call directly and return result

    Args:
        handler: The callable to invoke
        arg: The argument to pass to the handler
        loop: Optional event loop (if running from a thread)
    """
    if inspect.iscoroutinefunction(handler):
        coro = handler(arg)  # type: ignore
        if loop:
            # Threaded context: fire and forget (or return future)
            # run_coroutine_threadsafe returns a concurrent.futures.Future
            return asyncio.run_coroutine_threadsafe(coro, loop)
        else:
            # Async context: return coro to be awaited
            return coro  # type: ignore
    else:
        # Sync handler
        return handler(arg)  # type: ignore
