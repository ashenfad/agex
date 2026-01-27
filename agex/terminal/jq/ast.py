"""AST nodes for jq filter expressions."""

from dataclasses import dataclass
from typing import Any


@dataclass
class Identity:
    """`.` - pass input through unchanged."""

    pass


@dataclass
class RecursiveDescent:
    """`..` - recursively descend into all values."""

    pass


@dataclass
class Field:
    """`.foo` - access object field."""

    name: str


@dataclass
class OptionalField:
    """`.foo?` - access field, suppress errors if missing."""

    name: str


@dataclass
class Index:
    """.[n] - access array element by index."""

    index: int


@dataclass
class Slice:
    """.[start:end] - array slice."""

    start: int | None
    end: int | None


@dataclass
class Iterate:
    """.[] - iterate over array elements or object values."""

    pass


@dataclass
class Pipe:
    """left | right - pipe output of left into right."""

    left: "Expr"
    right: "Expr"


@dataclass
class Literal:
    """A literal value (string, number, bool, null)."""

    value: Any


@dataclass
class FunctionCall:
    """Function call like `keys`, `length`, `select(expr)`."""

    name: str
    args: list["Expr"]


@dataclass
class Comparison:
    """Comparison: ==, !=, <, >, <=, >=."""

    op: str
    left: "Expr"
    right: "Expr"


@dataclass
class BoolOp:
    """Boolean operation: and, or."""

    op: str  # "and" or "or"
    left: "Expr"
    right: "Expr"


@dataclass
class Not:
    """Boolean not."""

    expr: "Expr"


@dataclass
class Arithmetic:
    """Arithmetic: +, -, *, /, %."""

    op: str
    left: "Expr"
    right: "Expr"


@dataclass
class Negative:
    """Unary negation: -expr."""

    expr: "Expr"


@dataclass
class ArrayConstruct:
    """[expr] - collect results into array."""

    expr: "Expr | None"  # None for empty array []


@dataclass
class ObjectConstruct:
    """{key: value, ...} - construct object."""

    entries: list[tuple["Expr | str", "Expr"]]  # key can be expr or literal string


@dataclass
class Conditional:
    """if cond then expr else expr end."""

    cond: "Expr"
    then_expr: "Expr"
    else_expr: "Expr"


@dataclass
class Alternative:
    """expr // default - null coalescing."""

    left: "Expr"
    right: "Expr"


# Type alias for any expression
Expr = (
    Identity
    | RecursiveDescent
    | Field
    | OptionalField
    | Index
    | Slice
    | Iterate
    | Pipe
    | Literal
    | FunctionCall
    | Comparison
    | BoolOp
    | Not
    | Arithmetic
    | Negative
    | ArrayConstruct
    | ObjectConstruct
    | Conditional
    | Alternative
)
