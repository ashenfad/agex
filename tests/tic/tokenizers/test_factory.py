import pytest

from tic.tokenizers import TiktokenTokenizer, get_tokenizer


def test_get_tokenizer_for_openai_model():
    """Tests that a known gpt model returns a TiktokenTokenizer."""
    tokenizer = get_tokenizer("gpt-4o")
    assert isinstance(tokenizer, TiktokenTokenizer)
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_get_tokenizer_unsupported_model():
    """Tests that an unsupported model raises a ValueError."""
    with pytest.raises(ValueError, match="No tokenizer available"):
        get_tokenizer("claude-3-opus-20240229")
