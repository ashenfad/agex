from typing import List, Protocol


class Tokenizer(Protocol):
    """
    A common interface for tokenizers, ensuring compatibility between different
    providers like tiktoken and Hugging Face tokenizers.
    """

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs."""
        ...

    def decode(self, tokens: List[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        ...
