from .core import Tokenizer

__all__ = ["Tokenizer", "TiktokenTokenizer", "get_tokenizer"]


def __getattr__(name: str):
    """Lazy attribute access for ``TiktokenTokenizer``.

    Keeps ``from agex.tokenizers import TiktokenTokenizer`` working
    while deferring the tiktoken import (which is non-trivial) until
    the name is actually requested.
    """
    if name == "TiktokenTokenizer":
        from .tiktoken import TiktokenTokenizer  # noqa: PLC0415

        return TiktokenTokenizer
    raise AttributeError(f"module 'agex.tokenizers' has no attribute {name!r}")


class _ApproxTokenizer:
    """Rough char-based tokenizer for environments without tiktoken."""

    def encode(self, text: str) -> list[int]:
        # ~4 chars per token is a common approximation.
        return list(range(max(1, len(text) // 4)))

    def decode(self, tokens: list[int]) -> str:
        return ""


# Probe result of importing tiktoken: True = available, False = not
# available, None = not yet probed. Lazy so ``import agex`` stays fast
# when no tokenization is needed.
_has_tiktoken: bool | None = None


def get_tokenizer(model_name: str) -> Tokenizer:
    """Factory function to get the appropriate tokenizer for a given model name.

    Tiktoken is imported on first call rather than at module load, so
    ``import agex`` doesn't pay its initialization cost up front.
    """
    global _has_tiktoken
    if _has_tiktoken is None:
        try:
            from .tiktoken import TiktokenTokenizer  # noqa: F401, PLC0415

            _has_tiktoken = True
        except Exception:
            # tiktoken unavailable (e.g. Pyodide — encoding files blocked
            # by CORS) or any other init failure. Fall back gracefully.
            _has_tiktoken = False

    if not _has_tiktoken:
        return _ApproxTokenizer()  # type: ignore[return-value]

    from .tiktoken import TiktokenTokenizer  # noqa: PLC0415

    return TiktokenTokenizer("gpt-4")
