from .core import Tokenizer

try:
    from .tiktoken import TiktokenTokenizer

    _has_tiktoken = True
except (ImportError, Exception):
    # tiktoken unavailable (e.g. Pyodide — encoding files blocked by CORS)
    _has_tiktoken = False


class _ApproxTokenizer:
    """Rough char-based tokenizer for environments without tiktoken."""

    def encode(self, text: str) -> list[int]:
        # ~4 chars per token is a common approximation
        return list(range(max(1, len(text) // 4)))

    def decode(self, tokens: list[int]) -> str:
        return ""


def get_tokenizer(model_name: str) -> Tokenizer:
    """
    Factory function to get the appropriate tokenizer for a given model name.
    """
    if not _has_tiktoken:
        return _ApproxTokenizer()  # type: ignore[return-value]
    return TiktokenTokenizer("gpt-4")
