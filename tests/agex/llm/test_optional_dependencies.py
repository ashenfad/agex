import os
from unittest.mock import patch

from agex import connect_llm


def test_openai_import_error():
    # Test that calling get_llm_client with provider="openai" raises an ImportError
    # if the 'openai' package is not installed.
    with patch("agex.llm.OpenAIClient", None):
        with patch.dict(os.environ, {"AGEX_LLM_PROVIDER": "openai"}, clear=True):
            try:
                connect_llm(provider="openai")
                assert False, "Expected ImportError to be raised"
            except ImportError as e:
                assert 'pip install "agex[openai]"' in str(e)


def test_anthropic_import_error():
    # Test that calling get_llm_client with provider="anthropic" raises an ImportError
    # if the 'anthropic' package is not installed.
    with patch("agex.llm.AnthropicClient", None):
        with patch.dict(os.environ, {"AGEX_LLM_PROVIDER": "anthropic"}, clear=True):
            try:
                connect_llm(provider="anthropic")
                assert False, "Expected ImportError to be raised"
            except ImportError as e:
                assert 'pip install "agex[anthropic]"' in str(e)


def test_gemini_import_error():
    # Test that calling get_llm_client with provider="gemini" raises an ImportError
    # if the 'google-generativeai' package is not installed.
    with patch("agex.llm.GeminiClient", None):
        with patch.dict(os.environ, {"AGEX_LLM_PROVIDER": "gemini"}, clear=True):
            try:
                connect_llm(provider="gemini")
                assert False, "Expected ImportError to be raised"
            except ImportError as e:
                assert 'pip install "agex[gemini]"' in str(e)


def test_dummy_client_always_available():
    # Test that the dummy client is always available, even if other dependencies are not installed.
    with patch("agex.llm.OpenAIClient", None):
        client = connect_llm(provider="dummy")
        assert client is not None
        assert client.__class__.__name__ == "DummyLLMClient"


def test_unsupported_provider_error():
    # Test that a ValueError is raised for an unsupported provider.
    with patch.dict(os.environ, {}, clear=True):
        try:
            connect_llm(provider="invalid")  # type: ignore
            assert False, "Expected ValueError to be raised"
        except ValueError as e:
            assert "Unsupported provider" in str(e)
            assert "invalid" in str(e)
            assert "dummy" in str(e)  # Assuming dummy is always available


def test_available_providers_in_error_message():
    # Test that the error message for an unsupported provider lists the currently available providers.
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}, clear=True):
        # Only dummy is available
        with (
            patch("agex.llm.OpenAIClient", None),
            patch("agex.llm.AnthropicClient", None),
            patch("agex.llm.GeminiClient", None),
        ):
            try:
                connect_llm(provider="invalid")  # type: ignore
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                error_str = str(e)
                assert "dummy" in error_str
                assert "openai" not in error_str
                assert "anthropic" not in error_str
                assert "gemini" not in error_str

        # Test with OpenAI available
        with (
            patch("agex.llm.AnthropicClient", None),
            patch("agex.llm.GeminiClient", None),
        ):
            try:
                connect_llm(provider="invalid")  # type: ignore
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                error_str = str(e)
                assert "openai" in error_str
                assert "anthropic" not in error_str
                assert "gemini" not in error_str

        # Test with Anthropic available
        with patch("agex.llm.OpenAIClient", None), patch("agex.llm.GeminiClient", None):
            try:
                connect_llm(provider="invalid")  # type: ignore
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                error_str = str(e)
                assert "anthropic" in error_str
                assert "openai" not in error_str

        # Test with Gemini available
        with (
            patch("agex.llm.OpenAIClient", None),
            patch("agex.llm.AnthropicClient", None),
        ):
            try:
                connect_llm(provider="invalid")  # type: ignore
                assert False, "Expected ValueError to be raised"
            except ValueError as e:
                error_str = str(e)
                assert "gemini" in error_str
                assert "openai" not in error_str
