"""Recursive descent parser for jq filter expressions."""

from dataclasses import dataclass

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


class ParseError(Exception):
    """Error parsing jq filter expression."""

    pass


def _error_context(text: str, pos: int, width: int = 20) -> str:
    """Generate context string showing where error occurred."""
    start = max(0, pos - width)
    end = min(len(text), pos + width)
    before = text[start:pos]
    after = text[pos:end]
    marker = "^"
    if start > 0:
        before = "..." + before
    if end < len(text):
        after = after + "..."
    return f"{before}{marker}{after}"


@dataclass
class Token:
    type: str
    value: str
    pos: int


class Lexer:
    """Tokenize jq filter expression."""

    KEYWORDS = {
        "and",
        "or",
        "not",
        "if",
        "then",
        "else",
        "end",
        "as",
        "true",
        "false",
        "null",
    }

    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.tokens: list[Token] = []
        self._tokenize()

    def _tokenize(self):
        while self.pos < len(self.text):
            # Skip whitespace
            if self.text[self.pos].isspace():
                self.pos += 1
                continue

            # Two-character operators
            if self.pos + 1 < len(self.text):
                two = self.text[self.pos : self.pos + 2]
                if two in ("==", "!=", "<=", ">=", "//", ".."):
                    self.tokens.append(Token(two, two, self.pos))
                    self.pos += 2
                    continue

            # Single character tokens
            ch = self.text[self.pos]
            if ch in ".[]{}()|,;:<>+-*/%?":
                self.tokens.append(Token(ch, ch, self.pos))
                self.pos += 1
                continue

            # String literal
            if ch == '"':
                start = self.pos
                self.pos += 1
                value = ""
                while self.pos < len(self.text) and self.text[self.pos] != '"':
                    if self.text[self.pos] == "\\" and self.pos + 1 < len(self.text):
                        next_ch = self.text[self.pos + 1]
                        if next_ch == "n":
                            value += "\n"
                        elif next_ch == "t":
                            value += "\t"
                        elif next_ch == "\\":
                            value += "\\"
                        elif next_ch == '"':
                            value += '"'
                        else:
                            value += next_ch
                        self.pos += 2
                    else:
                        value += self.text[self.pos]
                        self.pos += 1
                if self.pos >= len(self.text):
                    ctx = _error_context(self.text, start)
                    raise ParseError(f'Unterminated string (missing closing "): {ctx}')
                self.pos += 1  # skip closing quote
                self.tokens.append(Token("STRING", value, start))
                continue

            # Number
            if ch.isdigit() or (
                ch == "-"
                and self.pos + 1 < len(self.text)
                and self.text[self.pos + 1].isdigit()
            ):
                start = self.pos
                if ch == "-":
                    self.pos += 1
                while self.pos < len(self.text) and self.text[self.pos].isdigit():
                    self.pos += 1
                if self.pos < len(self.text) and self.text[self.pos] == ".":
                    self.pos += 1
                    while self.pos < len(self.text) and self.text[self.pos].isdigit():
                        self.pos += 1
                value = self.text[start : self.pos]
                self.tokens.append(Token("NUMBER", value, start))
                continue

            # Identifier or keyword
            if ch.isalpha() or ch == "_":
                start = self.pos
                while self.pos < len(self.text) and (
                    self.text[self.pos].isalnum() or self.text[self.pos] == "_"
                ):
                    self.pos += 1
                value = self.text[start : self.pos]
                if value in self.KEYWORDS:
                    self.tokens.append(Token(value.upper(), value, start))
                else:
                    self.tokens.append(Token("IDENT", value, start))
                continue

            # Variable $name
            if ch == "$":
                start = self.pos
                self.pos += 1
                while self.pos < len(self.text) and (
                    self.text[self.pos].isalnum() or self.text[self.pos] == "_"
                ):
                    self.pos += 1
                value = self.text[start : self.pos]
                self.tokens.append(Token("VAR", value, start))
                continue

            ctx = _error_context(self.text, self.pos)
            raise ParseError(f"Unexpected character '{ch}': {ctx}")

        self.tokens.append(Token("EOF", "", self.pos))


class Parser:
    """Parse jq filter expression into AST."""

    def __init__(self, text: str):
        self.text = text
        self.lexer = Lexer(text)
        self.tokens = self.lexer.tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # EOF

    def consume(self, expected: str | None = None) -> Token:
        tok = self.peek()
        if expected and tok.type != expected:
            ctx = _error_context(self.text, tok.pos)
            raise ParseError(f"Expected {expected}, got {tok.type}: {ctx}")
        self.pos += 1
        return tok

    def match(self, *types: str) -> bool:
        return self.peek().type in types

    def parse(self) -> Expr:
        expr = self.parse_pipe()
        if not self.match("EOF"):
            tok = self.peek()
            ctx = _error_context(self.text, tok.pos)
            raise ParseError(f"Unexpected token {tok.type}: {ctx}")
        return expr

    def parse_pipe(self) -> Expr:
        """Parse pipe: expr | expr | ..."""
        left = self.parse_alternative()
        while self.match("|"):
            self.consume("|")
            right = self.parse_alternative()
            left = Pipe(left, right)
        return left

    def parse_alternative(self) -> Expr:
        """Parse alternative: expr // expr"""
        left = self.parse_or()
        while self.match("//"):
            self.consume("//")
            right = self.parse_or()
            left = Alternative(left, right)
        return left

    def parse_or(self) -> Expr:
        """Parse or: expr or expr"""
        left = self.parse_and()
        while self.match("OR"):
            self.consume("OR")
            right = self.parse_and()
            left = BoolOp("or", left, right)
        return left

    def parse_and(self) -> Expr:
        """Parse and: expr and expr"""
        left = self.parse_not()
        while self.match("AND"):
            self.consume("AND")
            right = self.parse_not()
            left = BoolOp("and", left, right)
        return left

    def parse_not(self) -> Expr:
        """Parse not: not expr (prefix form, rarely used in jq)"""
        # In jq, 'not' is primarily used as a filter: expr | not
        # But we also support prefix form for compatibility: not expr
        if self.match("NOT"):
            # Check if next token starts an expression (not just EOF or operator)
            if self.peek(1).type in (
                ".",
                "(",
                "[",
                "{",
                "NUMBER",
                "STRING",
                "IDENT",
                "TRUE",
                "FALSE",
                "NULL",
                "IF",
            ):
                self.consume("NOT")
                return Not(self.parse_not())
            # Otherwise treat 'not' as an identifier (function call)
        return self.parse_comparison()

    def parse_comparison(self) -> Expr:
        """Parse comparison: expr == expr, etc."""
        left = self.parse_additive()
        if self.match("==", "!=", "<", ">", "<=", ">="):
            op = self.consume().value
            right = self.parse_additive()
            return Comparison(op, left, right)
        return left

    def parse_additive(self) -> Expr:
        """Parse addition/subtraction: expr + expr - expr"""
        left = self.parse_multiplicative()
        while self.match("+", "-"):
            op = self.consume().value
            right = self.parse_multiplicative()
            left = Arithmetic(op, left, right)
        return left

    def parse_multiplicative(self) -> Expr:
        """Parse multiplication/division: expr * expr / expr"""
        left = self.parse_unary()
        while self.match("*", "/", "%"):
            op = self.consume().value
            right = self.parse_unary()
            left = Arithmetic(op, left, right)
        return left

    def parse_unary(self) -> Expr:
        """Parse unary minus: -expr"""
        if self.match("-"):
            self.consume("-")
            return Negative(self.parse_unary())
        return self.parse_postfix()

    def parse_postfix(self) -> Expr:
        """Parse postfix operations: .foo, [n], [], ?, etc."""
        expr = self.parse_primary()

        while True:
            if self.match("."):
                self.consume(".")
                if self.match("IDENT"):
                    name = self.consume("IDENT").value
                    optional = False
                    if self.match("?"):
                        self.consume("?")
                        optional = True
                    # Chain: wrap in a pipe with field access
                    field = OptionalField(name) if optional else Field(name)
                    expr = Pipe(expr, field)
                elif self.match("["):
                    # .[...] - handled below
                    self.pos -= 1  # unconsume the dot, let bracket handler deal with it
                    # Actually, we need to handle .[] case
                    break
                else:
                    tok = self.peek()
                    ctx = _error_context(self.text, tok.pos)
                    raise ParseError(
                        f"Expected field name after '.', got {tok.type}: {ctx}"
                    )

            elif self.match("["):
                self.consume("[")
                if self.match("]"):
                    # .[] - iterate
                    self.consume("]")
                    expr = Pipe(expr, Iterate())
                elif self.match(":"):
                    # .[:n] - slice from start
                    self.consume(":")
                    end = (
                        int(self.consume("NUMBER").value)
                        if self.match("NUMBER")
                        else None
                    )
                    self.consume("]")
                    expr = Pipe(expr, Slice(None, end))
                elif self.match("NUMBER"):
                    num_tok = self.consume("NUMBER")
                    num = int(num_tok.value)
                    if self.match(":"):
                        # .[n:] or .[n:m]
                        self.consume(":")
                        end = (
                            int(self.consume("NUMBER").value)
                            if self.match("NUMBER")
                            else None
                        )
                        self.consume("]")
                        expr = Pipe(expr, Slice(num, end))
                    else:
                        # .[n]
                        self.consume("]")
                        expr = Pipe(expr, Index(num))
                elif self.match("-"):
                    # .[-n]
                    self.consume("-")
                    num = -int(self.consume("NUMBER").value)
                    self.consume("]")
                    expr = Pipe(expr, Index(num))
                else:
                    tok = self.peek()
                    ctx = _error_context(self.text, tok.pos)
                    raise ParseError(
                        f"Expected number or ':' in index, got {tok.type}: {ctx}. "
                        f"Examples: .[0], .[-1], .[2:5], .[:3]"
                    )

            elif self.match("?"):
                # Optional - convert last field to optional
                self.consume("?")
                if isinstance(expr, Pipe) and isinstance(expr.right, Field):
                    expr = Pipe(expr.left, OptionalField(expr.right.name))
                # Otherwise ignore for now

            else:
                break

        return expr

    def parse_primary(self) -> Expr:
        """Parse primary expressions."""
        tok = self.peek()

        # Identity: .
        if tok.type == ".":
            self.consume(".")
            if self.match("."):
                # .. recursive descent
                self.consume(".")
                return RecursiveDescent()
            if self.match("IDENT"):
                # .foo
                name = self.consume("IDENT").value
                optional = False
                if self.match("?"):
                    self.consume("?")
                    optional = True
                return OptionalField(name) if optional else Field(name)
            if self.match("["):
                # .[...] - iterate or index
                self.consume("[")
                if self.match("]"):
                    self.consume("]")
                    return Iterate()
                elif self.match(":"):
                    self.consume(":")
                    end = (
                        int(self.consume("NUMBER").value)
                        if self.match("NUMBER")
                        else None
                    )
                    self.consume("]")
                    return Slice(None, end)
                elif self.match("NUMBER"):
                    num = int(self.consume("NUMBER").value)
                    if self.match(":"):
                        self.consume(":")
                        end = (
                            int(self.consume("NUMBER").value)
                            if self.match("NUMBER")
                            else None
                        )
                        self.consume("]")
                        return Slice(num, end)
                    self.consume("]")
                    return Index(num)
                elif self.match("-"):
                    self.consume("-")
                    num = -int(self.consume("NUMBER").value)
                    self.consume("]")
                    return Index(num)
                else:
                    tok = self.peek()
                    ctx = _error_context(self.text, tok.pos)
                    raise ParseError(
                        f"Expected number in index, got {tok.type}: {ctx}. "
                        f"Examples: .[0], .[1:3]"
                    )
            return Identity()

        # Parenthesized expression
        if tok.type == "(":
            self.consume("(")
            expr = self.parse_pipe()
            self.consume(")")
            return expr

        # Array construction: [expr] or []
        if tok.type == "[":
            self.consume("[")
            if self.match("]"):
                self.consume("]")
                return ArrayConstruct(None)
            expr = self.parse_pipe()
            self.consume("]")
            return ArrayConstruct(expr)

        # Object construction: {key: value, ...}
        if tok.type == "{":
            return self.parse_object()

        # Literals
        if tok.type == "NUMBER":
            self.consume("NUMBER")
            if "." in tok.value:
                return Literal(float(tok.value))
            return Literal(int(tok.value))

        if tok.type == "STRING":
            self.consume("STRING")
            return Literal(tok.value)

        if tok.type == "TRUE":
            self.consume("TRUE")
            return Literal(True)

        if tok.type == "FALSE":
            self.consume("FALSE")
            return Literal(False)

        if tok.type == "NULL":
            self.consume("NULL")
            return Literal(None)

        # Conditional: if cond then expr else expr end
        if tok.type == "IF":
            self.consume("IF")
            cond = self.parse_pipe()
            self.consume("THEN")
            then_expr = self.parse_pipe()
            self.consume("ELSE")
            else_expr = self.parse_pipe()
            self.consume("END")
            return Conditional(cond, then_expr, else_expr)

        # Function call or bare identifier
        if tok.type == "IDENT":
            name = self.consume("IDENT").value
            args: list[Expr] = []
            if self.match("("):
                self.consume("(")
                if not self.match(")"):
                    args.append(self.parse_pipe())
                    while self.match(";"):
                        self.consume(";")
                        args.append(self.parse_pipe())
                self.consume(")")
            return FunctionCall(name, args)

        # Handle 'not' as a filter function when used standalone (e.g., "true | not")
        if tok.type == "NOT":
            self.consume("NOT")
            return FunctionCall("not", [])

        ctx = _error_context(self.text, tok.pos)
        raise ParseError(
            f"Unexpected token {tok.type}: {ctx}. "
            f"Expected: ., .field, [array], {{object}}, number, string, or function()"
        )

    def parse_object(self) -> ObjectConstruct:
        """Parse object construction: {key: value, ...}"""
        self.consume("{")
        entries: list[tuple[Expr | str, Expr]] = []

        if not self.match("}"):
            while True:
                # Key can be identifier, string, or expression in parens
                if self.match("IDENT"):
                    key_tok = self.consume("IDENT")
                    if self.match(":"):
                        self.consume(":")
                        value = self.parse_pipe()
                        entries.append((key_tok.value, value))
                    else:
                        # Shorthand: {foo} means {foo: .foo}
                        entries.append((key_tok.value, Field(key_tok.value)))
                elif self.match("STRING"):
                    key = self.consume("STRING").value
                    self.consume(":")
                    value = self.parse_pipe()
                    entries.append((key, value))
                elif self.match("("):
                    self.consume("(")
                    key_expr = self.parse_pipe()
                    self.consume(")")
                    self.consume(":")
                    value = self.parse_pipe()
                    entries.append((key_expr, value))
                else:
                    tok = self.peek()
                    ctx = _error_context(self.text, tok.pos)
                    raise ParseError(
                        f"Expected object key (identifier, string, or (expr)), "
                        f"got {tok.type}: {ctx}"
                    )

                if self.match(","):
                    self.consume(",")
                else:
                    break

        self.consume("}")
        return ObjectConstruct(entries)


def parse_filter(text: str) -> Expr:
    """Parse a jq filter expression string into an AST."""
    parser = Parser(text)
    return parser.parse()
