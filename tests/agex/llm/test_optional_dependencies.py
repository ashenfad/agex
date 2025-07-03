from unittest.mock import patch

import pytest

from agex.llm import get_llm_client


def test_missing_openai_dependency():
    """Test that missing OpenAI dependency raises helpful error message."""
    with patch("agex.llm.OpenAIClient", None):
        with pytest.raises(
            ImportError, match="OpenAI provider requires the 'openai' package"
        ):
            get_llm_client(provider="openai")


def test_missing_anthropic_dependency():
    """Test that missing Anthropic dependency raises helpful error message."""
    with patch("agex.llm.AnthropicClient", None):
        with pytest.raises(
            ImportError, match="Anthropic provider requires the 'anthropic' package"
        ):
            get_llm_client(provider="anthropic")


def test_missing_gemini_dependency():
    """Test that missing Gemini dependency raises helpful error message."""
    with patch("agex.llm.GeminiClient", None):
        with pytest.raises(
            ImportError,
            match="Gemini provider requires the 'google-generativeai' package",
        ):
            get_llm_client(provider="gemini")


def test_dummy_client_always_available():
    """Test that dummy client is always available regardless of optional dependencies."""
    # Should work even if we patch out all other clients
    with (
        patch("agex.llm.OpenAIClient", None),
        patch("agex.llm.AnthropicClient", None),
        patch("agex.llm.GeminiClient", None),
    ):
        client = get_llm_client(provider="dummy")
        assert client is not None
        assert client.model == "dummy"


def test_available_providers_list_updates():
    """Test that the error message shows only available providers."""
    # Patch out OpenAI and Gemini, leave Anthropic
    with patch("agex.llm.OpenAIClient", None), patch("agex.llm.GeminiClient", None):
        with pytest.raises(ValueError) as exc_info:
            get_llm_client(provider="invalid")  # type: ignore

        error_message = str(exc_info.value)
        assert "Available providers are: dummy, anthropic" in error_message
        assert "openai" not in error_message
        assert "gemini" not in error_message


def test_all_providers_missing_except_dummy():
    """Test behavior when all optional providers are missing."""
    with (
        patch("agex.llm.OpenAIClient", None),
        patch("agex.llm.AnthropicClient", None),
        patch("agex.llm.GeminiClient", None),
    ):
        with pytest.raises(ValueError) as exc_info:
            get_llm_client(provider="invalid")  # type: ignore

        error_message = str(exc_info.value)
        assert "Available providers are: dummy" in error_message


def test_installation_instructions_in_error_messages():
    """Test that error messages include installation instructions."""
    with patch("agex.llm.OpenAIClient", None):
        with pytest.raises(ImportError) as exc_info:
            get_llm_client(provider="openai")

        error_message = str(exc_info.value)
        assert "pip install 'agex[openai]'" in error_message

    with patch("agex.llm.AnthropicClient", None):
        with pytest.raises(ImportError) as exc_info:
            get_llm_client(provider="anthropic")

        error_message = str(exc_info.value)
        assert "pip install 'agex[anthropic]'" in error_message

    with patch("agex.llm.GeminiClient", None):
        with pytest.raises(ImportError) as exc_info:
            get_llm_client(provider="gemini")

        error_message = str(exc_info.value)
        assert "pip install 'agex[gemini]'" in error_message
