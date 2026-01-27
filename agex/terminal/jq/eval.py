"""Evaluate jq filter expressions against JSON data."""

from collections.abc import Iterator
from typing import Any

from .ast import (
    Alternative,
    Arithmetic,
    ArrayConstruct,
    BoolOp,
    Comparison,
    Conditional,
    Expr,
    Field,
    FunctionCall,
    Identity,
    Index,
    Iterate,
    Literal,
    Negative,
    Not,
    ObjectConstruct,
    OptionalField,
    Pipe,
    RecursiveDescent,
    Slice,
)


class JqError(Exception):
    """Error during jq evaluation."""

    pass


class JqTypeError(JqError):
    """Type error during jq evaluation."""

    pass


def evaluate(expr: Expr, data: Any) -> Iterator[Any]:
    """Evaluate a jq expression against input data, yielding results.

    jq expressions can produce multiple outputs (e.g., .[] iterates),
    so this is a generator.
    """
    match expr:
        case Identity():
            yield data

        case RecursiveDescent():
            yield from _recursive_descent(data)

        case Field(name=name):
            if data is None:
                yield None
            elif isinstance(data, dict):
                if name in data:
                    yield data[name]
                else:
                    available = list(data.keys())[:5]
                    hint = f" (available keys: {available})" if available else ""
                    raise JqError(
                        f"Object does not have key '{name}'{hint}. "
                        f"Use .{name}? for optional access"
                    )
            else:
                raise JqTypeError(
                    f"Cannot index {type(data).__name__} with .{name} "
                    f"(expected object, got {_jq_type(data)})"
                )

        case OptionalField(name=name):
            if data is None:
                yield None
            elif isinstance(data, dict):
                yield data.get(name)
            else:
                yield None  # Suppress error for optional

        case Index(index=idx):
            if data is None:
                yield None
            elif isinstance(data, list):
                if -len(data) <= idx < len(data):
                    yield data[idx]
                else:
                    yield None  # Out of bounds returns null
            elif isinstance(data, str):
                if -len(data) <= idx < len(data):
                    yield data[idx]
                else:
                    yield None
            else:
                raise JqTypeError(f"Cannot index {type(data).__name__} with number")

        case Slice(start=start, end=end):
            if data is None:
                yield None
            elif isinstance(data, (list, str)):
                yield data[start:end]
            else:
                raise JqTypeError(f"Cannot slice {type(data).__name__}")

        case Iterate():
            if data is None:
                pass  # No output
            elif isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                yield from data.values()
            else:
                raise JqTypeError(
                    f"Cannot iterate with .[] over {_jq_type(data)} "
                    f"(expected array or object)"
                )

        case Pipe(left=left, right=right):
            for intermediate in evaluate(left, data):
                yield from evaluate(right, intermediate)

        case Literal(value=value):
            yield value

        case FunctionCall(name=name, args=args):
            yield from _call_function(name, args, data)

        case Comparison(op=op, left=left, right=right):
            # For comparisons, we evaluate both sides and compare
            # This assumes single outputs from each side
            left_vals = list(evaluate(left, data))
            right_vals = list(evaluate(right, data))
            if len(left_vals) != 1 or len(right_vals) != 1:
                raise JqError("Comparison operands must produce single values")
            lv, rv = left_vals[0], right_vals[0]
            yield _compare(op, lv, rv)

        case BoolOp(op=op, left=left, right=right):
            left_vals = list(evaluate(left, data))
            if len(left_vals) != 1:
                raise JqError("Boolean operand must produce single value")
            lv = left_vals[0]

            if op == "and":
                if not _is_truthy(lv):
                    yield False
                else:
                    right_vals = list(evaluate(right, data))
                    if len(right_vals) != 1:
                        raise JqError("Boolean operand must produce single value")
                    yield _is_truthy(right_vals[0])
            else:  # or
                if _is_truthy(lv):
                    yield True
                else:
                    right_vals = list(evaluate(right, data))
                    if len(right_vals) != 1:
                        raise JqError("Boolean operand must produce single value")
                    yield _is_truthy(right_vals[0])

        case Not(expr=inner):
            for val in evaluate(inner, data):
                yield not _is_truthy(val)

        case Arithmetic(op=op, left=left, right=right):
            left_vals = list(evaluate(left, data))
            right_vals = list(evaluate(right, data))
            if len(left_vals) != 1 or len(right_vals) != 1:
                raise JqError("Arithmetic operands must produce single values")
            yield _arithmetic(op, left_vals[0], right_vals[0])

        case Negative(expr=inner):
            for val in evaluate(inner, data):
                if isinstance(val, (int, float)):
                    yield -val
                else:
                    raise JqTypeError(f"Cannot negate {type(val).__name__}")

        case ArrayConstruct(expr=inner):
            if inner is None:
                yield []
            else:
                yield list(evaluate(inner, data))

        case ObjectConstruct(entries=entries):
            result = {}
            for key, value_expr in entries:
                # Evaluate key if it's an expression
                if isinstance(key, str):
                    key_str = key
                else:
                    key_vals = list(evaluate(key, data))
                    if len(key_vals) != 1:
                        raise JqError("Object key must produce single value")
                    key_str = str(key_vals[0])

                # Evaluate value
                value_vals = list(evaluate(value_expr, data))
                if len(value_vals) != 1:
                    raise JqError("Object value must produce single value")
                result[key_str] = value_vals[0]
            yield result

        case Conditional(cond=cond, then_expr=then_expr, else_expr=else_expr):
            cond_vals = list(evaluate(cond, data))
            if len(cond_vals) != 1:
                raise JqError("Condition must produce single value")
            if _is_truthy(cond_vals[0]):
                yield from evaluate(then_expr, data)
            else:
                yield from evaluate(else_expr, data)

        case Alternative(left=left, right=right):
            try:
                left_vals = list(evaluate(left, data))
                # If we got results and they're not all null/false, use them
                if left_vals and any(
                    v is not None and v is not False for v in left_vals
                ):
                    yield from left_vals
                else:
                    yield from evaluate(right, data)
            except JqError:
                yield from evaluate(right, data)

        case _:
            raise JqError(f"Unknown expression type: {type(expr)}")


def _recursive_descent(data: Any) -> Iterator[Any]:
    """Recursively yield all values in a nested structure."""
    yield data
    if isinstance(data, dict):
        for v in data.values():
            yield from _recursive_descent(v)
    elif isinstance(data, list):
        for item in data:
            yield from _recursive_descent(item)


def _is_truthy(value: Any) -> bool:
    """jq truthiness: false and null are falsy, everything else is truthy."""
    return value is not False and value is not None


def _compare(op: str, left: Any, right: Any) -> bool:
    """Perform comparison."""
    match op:
        case "==":
            return left == right
        case "!=":
            return left != right
        case "<":
            return left < right
        case ">":
            return left > right
        case "<=":
            return left <= right
        case ">=":
            return left >= right
        case _:
            raise JqError(f"Unknown comparison operator: {op}")


def _arithmetic(op: str, left: Any, right: Any) -> Any:
    """Perform arithmetic operation."""
    # String/array concatenation with +
    if op == "+" and isinstance(left, str) and isinstance(right, str):
        return left + right
    if op == "+" and isinstance(left, list) and isinstance(right, list):
        return left + right
    if op == "+" and isinstance(left, dict) and isinstance(right, dict):
        return {**left, **right}

    # Numeric operations
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        raise JqTypeError(
            f"Cannot perform {op} on {type(left).__name__} and {type(right).__name__}"
        )

    match op:
        case "+":
            return left + right
        case "-":
            return left - right
        case "*":
            return left * right
        case "/":
            if right == 0:
                raise JqError("Division by zero")
            return left / right
        case "%":
            if right == 0:
                raise JqError("Modulo by zero")
            return left % right
        case _:
            raise JqError(f"Unknown arithmetic operator: {op}")


def _call_function(name: str, args: list[Expr], data: Any) -> Iterator[Any]:
    """Call a built-in jq function."""
    match name:
        case "keys":
            if isinstance(data, dict):
                yield sorted(data.keys())
            elif isinstance(data, list):
                yield list(range(len(data)))
            else:
                raise JqTypeError(f"Cannot get keys of {type(data).__name__}")

        case "keys_unsorted":
            if isinstance(data, dict):
                yield list(data.keys())
            elif isinstance(data, list):
                yield list(range(len(data)))
            else:
                raise JqTypeError(f"Cannot get keys of {type(data).__name__}")

        case "values":
            if isinstance(data, dict):
                yield from data.values()
            elif isinstance(data, list):
                yield from data
            else:
                raise JqTypeError(f"Cannot get values of {type(data).__name__}")

        case "length":
            if data is None:
                yield 0
            elif isinstance(data, (str, list, dict)):
                yield len(data)
            else:
                raise JqTypeError(f"Cannot get length of {type(data).__name__}")

        case "type":
            yield _jq_type(data)

        case "empty":
            pass  # Produces no output

        case "null":
            yield None

        case "true":
            yield True

        case "false":
            yield False

        case "not":
            yield not _is_truthy(data)

        case "select":
            if len(args) != 1:
                raise JqError("select() requires exactly one argument")
            cond_vals = list(evaluate(args[0], data))
            if len(cond_vals) != 1:
                raise JqError("select() condition must produce single value")
            if _is_truthy(cond_vals[0]):
                yield data

        case "map":
            if len(args) != 1:
                raise JqError("map() requires exactly one argument (e.g., map(.name))")
            if not isinstance(data, list):
                raise JqTypeError(
                    f"map() requires array input, got {_jq_type(data)}. "
                    f"Use select() to filter or .[] to iterate"
                )
            result = []
            for item in data:
                result.extend(evaluate(args[0], item))
            yield result

        case "map_values":
            if len(args) != 1:
                raise JqError("map_values() requires exactly one argument")
            if isinstance(data, dict):
                result = {}
                for k, v in data.items():
                    vals = list(evaluate(args[0], v))
                    if vals:
                        result[k] = vals[0]
                yield result
            elif isinstance(data, list):
                result = []
                for item in data:
                    vals = list(evaluate(args[0], item))
                    if vals:
                        result.append(vals[0])
                yield result
            else:
                raise JqTypeError(f"Cannot map_values over {type(data).__name__}")

        case "sort":
            if not isinstance(data, list):
                raise JqTypeError(
                    f"sort requires array input, got {_jq_type(data)}. "
                    f"Try wrapping in array: [.[] | ...] | sort"
                )
            yield sorted(data, key=_sort_key)

        case "sort_by":
            if len(args) != 1:
                raise JqError("sort_by() requires exactly one argument")
            if not isinstance(data, list):
                raise JqTypeError("sort_by requires array input")

            def key_fn(item: Any) -> Any:
                vals = list(evaluate(args[0], item))
                return _sort_key(vals[0] if vals else None)

            yield sorted(data, key=key_fn)

        case "group_by":
            if len(args) != 1:
                raise JqError("group_by() requires exactly one argument")
            if not isinstance(data, list):
                raise JqTypeError("group_by requires array input")

            groups: dict[Any, list[Any]] = {}
            for item in data:
                vals = list(evaluate(args[0], item))
                key = vals[0] if vals else None
                # Convert to hashable
                if isinstance(key, list):
                    key = tuple(key)
                elif isinstance(key, dict):
                    key = tuple(sorted(key.items()))
                groups.setdefault(key, []).append(item)
            yield list(groups.values())

        case "unique":
            if not isinstance(data, list):
                raise JqTypeError("unique requires array input")
            seen: list[Any] = []
            for item in data:
                if item not in seen:
                    seen.append(item)
            yield seen

        case "unique_by":
            if len(args) != 1:
                raise JqError("unique_by() requires exactly one argument")
            if not isinstance(data, list):
                raise JqTypeError("unique_by requires array input")
            seen_keys: list[Any] = []
            result: list[Any] = []
            for item in data:
                vals = list(evaluate(args[0], item))
                key = vals[0] if vals else None
                if key not in seen_keys:
                    seen_keys.append(key)
                    result.append(item)
            yield result

        case "reverse":
            if not isinstance(data, list):
                raise JqTypeError("reverse requires array input")
            yield list(reversed(data))

        case "flatten":
            depth = 1
            if args:
                depth_vals = list(evaluate(args[0], data))
                if depth_vals:
                    depth = depth_vals[0]
            if not isinstance(data, list):
                raise JqTypeError("flatten requires array input")
            yield _flatten(data, depth)

        case "first":
            if isinstance(data, list) and data:
                yield data[0]
            elif args:
                # first(expr) - first result of expr
                for val in evaluate(args[0], data):
                    yield val
                    break

        case "last":
            if isinstance(data, list) and data:
                yield data[-1]
            elif args:
                # last(expr) - last result of expr
                result = None
                for val in evaluate(args[0], data):
                    result = val
                if result is not None:
                    yield result

        case "nth":
            if len(args) < 1:
                raise JqError("nth() requires at least one argument")
            n_vals = list(evaluate(args[0], data))
            if not n_vals:
                raise JqError("nth() index must produce a value")
            n = n_vals[0]
            if len(args) > 1:
                # nth(n; expr)
                for i, val in enumerate(evaluate(args[1], data)):
                    if i == n:
                        yield val
                        break
            elif isinstance(data, list):
                if 0 <= n < len(data):
                    yield data[n]

        case "add":
            if not isinstance(data, list):
                raise JqTypeError("add requires array input")
            if not data:
                yield None
            else:
                result = data[0]
                for item in data[1:]:
                    result = _arithmetic("+", result, item)
                yield result

        case "min":
            if not isinstance(data, list):
                raise JqTypeError("min requires array input")
            if not data:
                yield None
            else:
                yield min(data, key=_sort_key)

        case "max":
            if not isinstance(data, list):
                raise JqTypeError("max requires array input")
            if not data:
                yield None
            else:
                yield max(data, key=_sort_key)

        case "min_by":
            if len(args) != 1:
                raise JqError("min_by() requires exactly one argument")
            if not isinstance(data, list) or not data:
                yield None
            else:

                def key_fn(item: Any) -> Any:
                    vals = list(evaluate(args[0], item))
                    return _sort_key(vals[0] if vals else None)

                yield min(data, key=key_fn)

        case "max_by":
            if len(args) != 1:
                raise JqError("max_by() requires exactly one argument")
            if not isinstance(data, list) or not data:
                yield None
            else:

                def key_fn(item: Any) -> Any:
                    vals = list(evaluate(args[0], item))
                    return _sort_key(vals[0] if vals else None)

                yield max(data, key=key_fn)

        case "join":
            if len(args) != 1:
                raise JqError("join() requires exactly one argument")
            if not isinstance(data, list):
                raise JqTypeError("join requires array input")
            sep_vals = list(evaluate(args[0], data))
            sep = str(sep_vals[0]) if sep_vals else ""
            yield sep.join(str(x) for x in data)

        case "split":
            if len(args) != 1:
                raise JqError("split() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("split requires string input")
            sep_vals = list(evaluate(args[0], data))
            sep = str(sep_vals[0]) if sep_vals else ""
            yield data.split(sep)

        case "test":
            if len(args) != 1:
                raise JqError("test() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("test requires string input")
            pattern_vals = list(evaluate(args[0], data))
            if not pattern_vals:
                raise JqError("test() pattern must produce a value")
            import re

            pattern = str(pattern_vals[0])
            yield bool(re.search(pattern, data))

        case "match":
            if len(args) < 1:
                raise JqError("match() requires at least one argument")
            if not isinstance(data, str):
                raise JqTypeError("match requires string input")
            import re

            pattern_vals = list(evaluate(args[0], data))
            pattern = str(pattern_vals[0]) if pattern_vals else ""
            m = re.search(pattern, data)
            if m:
                yield {
                    "offset": m.start(),
                    "length": m.end() - m.start(),
                    "string": m.group(),
                    "captures": [
                        {
                            "offset": g.start() if g else -1,
                            "length": len(g) if g else 0,
                            "string": g,
                            "name": None,
                        }
                        for g in m.groups()
                    ],
                }
            else:
                pass  # No output if no match

        case "contains":
            if len(args) != 1:
                raise JqError("contains() requires exactly one argument")
            other_vals = list(evaluate(args[0], data))
            other = other_vals[0] if other_vals else None
            yield _contains(data, other)

        case "inside":
            if len(args) != 1:
                raise JqError("inside() requires exactly one argument")
            other_vals = list(evaluate(args[0], data))
            other = other_vals[0] if other_vals else None
            yield _contains(other, data)

        case "has":
            if len(args) != 1:
                raise JqError("has() requires exactly one argument")
            key_vals = list(evaluate(args[0], data))
            key = key_vals[0] if key_vals else None
            if isinstance(data, dict):
                yield key in data
            elif isinstance(data, list) and isinstance(key, int):
                yield 0 <= key < len(data)
            else:
                yield False

        case "in":
            if len(args) != 1:
                raise JqError("in() requires exactly one argument")
            obj_vals = list(evaluate(args[0], data))
            obj = obj_vals[0] if obj_vals else None
            if isinstance(obj, dict):
                yield data in obj
            elif isinstance(obj, list) and isinstance(data, int):
                yield 0 <= data < len(obj)
            else:
                yield False

        case "to_entries":
            if not isinstance(data, dict):
                raise JqTypeError("to_entries requires object input")
            yield [{"key": k, "value": v} for k, v in data.items()]

        case "from_entries":
            if not isinstance(data, list):
                raise JqTypeError("from_entries requires array input")
            result = {}
            for entry in data:
                if isinstance(entry, dict):
                    k = entry.get("key", entry.get("k", entry.get("name")))
                    v = entry.get("value", entry.get("v"))
                    if k is not None:
                        result[k] = v
            yield result

        case "with_entries":
            if len(args) != 1:
                raise JqError("with_entries() requires exactly one argument")
            if not isinstance(data, dict):
                raise JqTypeError("with_entries requires object input")
            entries = [{"key": k, "value": v} for k, v in data.items()]
            new_entries = []
            for entry in entries:
                vals = list(evaluate(args[0], entry))
                new_entries.extend(vals)
            result = {}
            for entry in new_entries:
                if isinstance(entry, dict):
                    k = entry.get("key", entry.get("k", entry.get("name")))
                    v = entry.get("value", entry.get("v"))
                    if k is not None:
                        result[k] = v
            yield result

        case "ascii_downcase" | "downcase":
            if not isinstance(data, str):
                raise JqTypeError("ascii_downcase requires string input")
            yield data.lower()

        case "ascii_upcase" | "upcase":
            if not isinstance(data, str):
                raise JqTypeError("ascii_upcase requires string input")
            yield data.upper()

        case "ltrimstr":
            if len(args) != 1:
                raise JqError("ltrimstr() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("ltrimstr requires string input")
            prefix_vals = list(evaluate(args[0], data))
            prefix = str(prefix_vals[0]) if prefix_vals else ""
            yield data.removeprefix(prefix)

        case "rtrimstr":
            if len(args) != 1:
                raise JqError("rtrimstr() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("rtrimstr requires string input")
            suffix_vals = list(evaluate(args[0], data))
            suffix = str(suffix_vals[0]) if suffix_vals else ""
            yield data.removesuffix(suffix)

        case "startswith":
            if len(args) != 1:
                raise JqError("startswith() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("startswith requires string input")
            prefix_vals = list(evaluate(args[0], data))
            prefix = str(prefix_vals[0]) if prefix_vals else ""
            yield data.startswith(prefix)

        case "endswith":
            if len(args) != 1:
                raise JqError("endswith() requires exactly one argument")
            if not isinstance(data, str):
                raise JqTypeError("endswith requires string input")
            suffix_vals = list(evaluate(args[0], data))
            suffix = str(suffix_vals[0]) if suffix_vals else ""
            yield data.endswith(suffix)

        case "tostring":
            if isinstance(data, str):
                yield data
            else:
                import json

                yield json.dumps(data)

        case "tonumber":
            if isinstance(data, (int, float)):
                yield data
            elif isinstance(data, str):
                try:
                    if "." in data:
                        yield float(data)
                    else:
                        yield int(data)
                except ValueError:
                    raise JqError(f"Cannot convert '{data}' to number")
            else:
                raise JqTypeError(f"Cannot convert {type(data).__name__} to number")

        case "floor":
            if isinstance(data, (int, float)):
                import math

                yield math.floor(data)
            else:
                raise JqTypeError("floor requires number input")

        case "ceil":
            if isinstance(data, (int, float)):
                import math

                yield math.ceil(data)
            else:
                raise JqTypeError("ceil requires number input")

        case "round":
            if isinstance(data, (int, float)):
                yield round(data)
            else:
                raise JqTypeError("round requires number input")

        case "abs":
            if isinstance(data, (int, float)):
                yield abs(data)
            else:
                raise JqTypeError("abs requires number input")

        case "error":
            msg = "error"
            if args:
                msg_vals = list(evaluate(args[0], data))
                if msg_vals:
                    msg = str(msg_vals[0])
            raise JqError(msg)

        case "debug":
            import sys

            print(f"DEBUG: {data}", file=sys.stderr)
            yield data

        case _:
            common = [
                "keys",
                "length",
                "type",
                "select",
                "map",
                "sort",
                "unique",
                "first",
                "last",
                "add",
                "join",
                "split",
                "has",
                "to_entries",
                "from_entries",
            ]
            raise JqError(
                f"Unknown function: {name}(). " f"Common functions: {', '.join(common)}"
            )


def _jq_type(value: Any) -> str:
    """Return jq type name for a value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _sort_key(value: Any) -> tuple[int, Any]:
    """Generate a sort key for jq's sorting semantics.

    jq sorts: null < false < true < numbers < strings < arrays < objects
    """
    if value is None:
        return (0, 0)
    if value is False:
        return (1, 0)
    if value is True:
        return (2, 0)
    if isinstance(value, (int, float)):
        return (3, value)
    if isinstance(value, str):
        return (4, value)
    if isinstance(value, list):
        return (5, tuple(_sort_key(x) for x in value))
    if isinstance(value, dict):
        return (6, tuple(sorted((k, _sort_key(v)) for k, v in value.items())))
    return (7, str(value))


def _flatten(data: list, depth: int) -> list:
    """Flatten a nested list up to specified depth."""
    if depth <= 0:
        return data
    result = []
    for item in data:
        if isinstance(item, list):
            result.extend(_flatten(item, depth - 1))
        else:
            result.append(item)
    return result


def _contains(a: Any, b: Any) -> bool:
    """Check if a contains b (jq semantics)."""
    if isinstance(b, dict):
        if not isinstance(a, dict):
            return False
        return all(k in a and _contains(a[k], v) for k, v in b.items())
    if isinstance(b, list):
        if not isinstance(a, list):
            return False
        return all(any(_contains(ai, bi) for ai in a) for bi in b)
    if isinstance(b, str) and isinstance(a, str):
        return b in a
    return a == b
