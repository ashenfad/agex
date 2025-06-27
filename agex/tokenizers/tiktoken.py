from typing import List

import tiktoken

from .core import Tokenizer


class TiktokenTokenizer(Tokenizer):
    """A tokenizer implementation using OpenAI's tiktoken library."""

    def __init__(self, model_name: str):
        self._encoding = tiktoken.encoding_for_model(model_name)

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs."""
        return self._encoding.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        return self._encoding.decode(tokens)
