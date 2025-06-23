from .core import Tokenizer
from .tiktoken import TiktokenTokenizer


def get_tokenizer(model_name: str) -> Tokenizer:
    """
    Factory function to get the appropriate tokenizer for a given model name.
    """
    # For now, we only support OpenAI models via tiktoken.
    # In the future, this can be extended for other providers.
    if model_name.startswith("gpt-"):
        return TiktokenTokenizer(model_name)
    else:
        raise ValueError(
            f"No tokenizer available for model '{model_name}'. "
            "Currently, only models starting with 'gpt-' are supported."
        )
