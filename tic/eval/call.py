import ast
from typing import Any, Callable

from ..agent import ExitClarify, ExitFail, ExitSuccess, _AgentExit
from .base import BaseEvaluator
from .error import EvalError
from .functions import UserFunction
from .objects import TicDataClass, TicObject
from .user_errors import (
    TicError,
    TicIndexError,
    TicKeyError,
    TicTypeError,
    TicValueError,
)

MAX_RANGE_SIZE = 10_000


# A simple placeholder object to act as the @dataclass decorator.
# Its only purpose is to be recognized by the evaluator.
class _DataclassDecorator:
    pass


dataclass = _DataclassDecorator()


def _tic_isinstance(obj: Any, class_or_tuple: Any) -> bool:
    """Custom isinstance function for the tic evaluator."""
    if isinstance(class_or_tuple, TicDataClass):
        if isinstance(obj, TicObject):
            return obj.cls is class_or_tuple
        return False
    # TODO: Handle tuple of types
    return isinstance(obj, class_or_tuple)


def _constrained_range(*args, **kwargs):
    """A wrapper around range() that enforces a maximum size."""
    if kwargs:
        raise TypeError("range() does not take keyword arguments.")
    r = range(*args)
    if len(r) > MAX_RANGE_SIZE:
        raise ValueError(f"Range exceeds maximum size of {MAX_RANGE_SIZE}")
    return list(r)


BUILTINS: dict[str, Callable[..., Any]] = {
    "len": len,
    "max": max,
    "min": min,
    "sum": sum,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "list": list,
    "abs": abs,
    "round": round,
    "all": all,
    "any": any,
    "sorted": sorted,
    "range": _constrained_range,
    "reversed": lambda x: list(reversed(x)),
    "zip": lambda *args: list(zip(*args)),
    "enumerate": lambda x: list(enumerate(x)),
    "map": lambda f, it: list(map(f, it)),
    "filter": lambda f, it: list(filter(f, it)),
    # Type introspection
    "isinstance": _tic_isinstance,
    "type": type,
    # Dataclasses
    "dataclass": dataclass,
    # User-level exceptions, mapped from Python's names
    "Exception": TicError,
    "ValueError": TicValueError,
    "TypeError": TicTypeError,
    "KeyError": TicKeyError,
    "IndexError": TicIndexError,
    # Agent exit signals
    "exit_success": ExitSuccess,
    "exit_fail": ExitFail,
    "exit_clarify": ExitClarify,
}

WHITELISTED_METHODS = {
    list: {
        "append",
        "clear",
        "copy",
        "count",
        "extend",
        "index",
        "insert",
        "pop",
        "remove",
        "reverse",
        "sort",
    },
    dict: {
        "clear",
        "copy",
        "get",
        "items",
        "keys",
        "pop",
        "setdefault",
        "update",
        "values",
    },
    set: {"add", "clear", "copy", "discard", "pop", "remove", "update"},
    str: {
        "upper",
        "lower",
        "strip",
        "split",
        "replace",
        "startswith",
        "endswith",
        "join",
    },
}

# Methods that return iterators/views and need to be materialized
MATERIALIZE_METHODS = {
    dict: {"keys": list, "values": list, "items": list},
}


class CallEvaluator(BaseEvaluator):
    """A mixin for evaluating function call nodes."""

    def visit_Call(self, node: ast.Call) -> Any:
        """Handles function calls."""
        # Common argument processing
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}

        # Case 1: Direct call by name (e.g., `my_func()`, `len()`)
        if isinstance(node.func, ast.Name):
            fn_name = node.func.id
            fn = BUILTINS.get(fn_name)
            if fn is None:
                fn = self.state.get(fn_name)

            if fn is None:
                raise EvalError(f"Function '{fn_name}' is not defined.", node)

            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code)

            if isinstance(fn, TicDataClass):
                return fn(*args, **kwargs)

            # It must be a builtin
            try:
                result = fn(*args, **kwargs)
                if isinstance(fn, type) and issubclass(fn, _AgentExit):
                    raise result
                return result
            except Exception as e:
                # Re-raise agent exit signals immediately
                if isinstance(e, _AgentExit):
                    raise e
                raise EvalError(
                    f"Error calling builtin function '{fn_name}': {e}", node, cause=e
                )

        # Case 2: Attribute call (e.g., `my_list.append()`)
        elif isinstance(node.func, ast.Attribute):
            obj = self.visit(node.func.value)
            method_name = node.func.attr
            obj_type = type(obj)

            allowed_methods = WHITELISTED_METHODS.get(obj_type)
            if not allowed_methods or method_name not in allowed_methods:
                raise EvalError(
                    f"Method '{method_name}' is not allowed on type '{obj_type.__name__}'.",
                    node,
                )

            method = getattr(obj, method_name)

            try:
                result = method(*args, **kwargs)
            except Exception as e:
                raise EvalError(
                    f"Error calling method '{method_name}': {e}", node, cause=e
                )

            materializer_map = MATERIALIZE_METHODS.get(obj_type, {})
            if materializer := materializer_map.get(method_name):
                return materializer(result)
            return result

        # Case 3: Indirect call (e.g., `get_func()()`)
        else:
            fn = self.visit(node.func)
            if isinstance(fn, UserFunction):
                return fn.execute(args, kwargs, self.source_code)

            if isinstance(fn, TicDataClass):
                return fn(*args, **kwargs)

            raise EvalError(
                f"Indirect call on a non-user function is not supported. Got: {type(fn).__name__}",
                node,
            )
