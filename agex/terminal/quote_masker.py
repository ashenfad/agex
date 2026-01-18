"""
Utility for masking quoted strings in shell commands.

This allows us to distinguish between quoted wildcards (literals) and unquoted
wildcards (glob patterns) after shlex tokenization.
"""

import re
import uuid
from typing import Dict, Tuple


def mask_quotes(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Replace all quoted substrings in text with unique mask tokens.

    Args:
        text: The raw shell command string.

    Returns:
        A tuple of (masked_text, mask_map).
        mask_map maps the token (e.g. __Q_A1B2_0__) back to the FULL original quoted string
        (including the quotes).
    """
    mask_map: Dict[str, str] = {}

    # Generate a unique ID for this masking session to avoid collisions
    session_id = uuid.uuid4().hex[:8]
    counter = 0

    # Regex to find quoted strings, handling escaped quotes inside.
    # See QuoteMasker class history for detailed regex explanation.
    # Use (?P=quote) for backreference to named group 'quote'.
    regex = r'(?<!\\)(?P<quote>["\'])(?P<content>(?:\\.|(?!(?P=quote)).)*)(?P=quote)'

    def replacer(match):
        nonlocal counter
        quote_char = match.group("quote")
        content = match.group("content")

        # The full quoted string
        original_quoted = quote_char + content + quote_char

        # Use concatenation to avoid f-string brace issues in tool processing
        token = "__Q_" + session_id + "_" + str(counter) + "__"
        counter += 1

        mask_map[token] = original_quoted
        return token

    # flags=re.DOTALL ensures '.' matches newlines if they appear in content
    masked_text = re.sub(regex, replacer, text, flags=re.DOTALL)

    return masked_text, mask_map


def unmask_quotes(text: str, mask_map: Dict[str, str]) -> str:
    """
    Restore masked tokens to their original quoted values.
    """
    for token, original in mask_map.items():
        text = text.replace(token, original)
    return text


def unmask_and_unquote(text: str, mask_map: Dict[str, str]) -> str:
    """
    Restore masked tokens but STRIP the outer quotes.
    This is used when the interpreter executes the command.
    """
    for token, original in mask_map.items():
        # original is like "foo" or 'bar'
        if len(original) >= 2:
            inner = original[1:-1]

            quote_char = original[0]
            if quote_char == '"':
                # Unescape \" -> "
                inner = inner.replace('"', '"')
            elif quote_char == "'":
                # Unescape \' -> '
                inner = inner.replace("'", "'")

            text = text.replace(token, inner)
        else:
            text = text.replace(token, original)

    return text
