import os
from unittest.mock import patch

from agex.llm import connect_llm


def test_dummy_provider_example():
    # Test the dummy provider example code from the documentation.
    # This should always work as it has no external dependencies.
    with patch.dict(os.environ, {}, clear=True):
        client = connect_llm(provider="dummy")
        assert client is not None
        assert client.__class__.__name__ == "DummyLLMClient"


def test_openai_provider_example():
    # Test the OpenAI provider example code from the documentation.
    # This test will pass if 'openai' is installed, and will be skipped otherwise.
    try:
        import openai  # noqa: F401
    except ImportError:
        # If openai is not installed, we can't test this, but that's expected.
        # The library should raise an ImportError if the user tries this without installation.
        return

    with patch.dict(
        os.environ,
        {
            "AGEX_LLM_PROVIDER": "openai",
            "AGEX_LLM_MODEL": "gpt-4.1-nano",
            "OPENAI_API_KEY": "test-key",
        },
        clear=True,
    ):
        client = connect_llm(provider="openai", model="gpt-4.1-nano")
        assert client is not None
        assert client.__class__.__name__ == "OpenAIClient"
        assert client.model == "gpt-4.1-nano"


def test_anthropic_provider_example():
    # Test the Anthropic provider example code from the documentation.
    # This test will pass if 'anthropic' is installed, and will be skipped otherwise.
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return

    with patch.dict(
        os.environ,
        {
            "AGEX_LLM_PROVIDER": "anthropic",
            "AGEX_LLM_MODEL": "claude-3-sonnet-20240229",
            "ANTHROPIC_API_KEY": "test-key",
        },
        clear=True,
    ):
        client = connect_llm(provider="anthropic", model="claude-3-sonnet-20240229")
        assert client is not None
        assert client.__class__.__name__ == "AnthropicClient"
        assert client.model == "claude-3-sonnet-20240229"


def test_gemini_provider_example():
    # Test the Gemini provider example code from the documentation.
    # This test will pass if 'google-generativeai' is installed, and will be skipped otherwise.
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        return

    with patch.dict(
        os.environ,
        {
            "AGEX_LLM_PROVIDER": "gemini",
            "AGEX_LLM_MODEL": "gemini-1.5-flash",
            "GOOGLE_API_KEY": "test-key",
        },
        clear=True,
    ):
        client = connect_llm(provider="gemini", model="gemini-1.5-flash")
        assert client is not None
        assert client.__class__.__name__ == "GeminiClient"
        assert client.model == "gemini-1.5-flash"


def test_llm_client_factory_example():
    # Test the LLM client factory example from the documentation.
    # This example shows how to create clients for different providers.
    # We use a loop and patch to simulate different installation scenarios.

    providers_to_test = {
        "openai": {"model": "gpt-4.1-nano"},
        "anthropic": {"model": "claude-3-sonnet-20240229"},
        "gemini": {"model": "gemini-1.5-flash"},
    }

    for provider, config in providers_to_test.items():
        try:
            # Simulate the required library being installed
            if provider == "openai":
                import openai  # noqa: F401
            elif provider == "anthropic":
                import anthropic  # noqa: F401
            elif provider == "gemini":
                import google.generativeai  # noqa: F401

            # Patch environment variables to avoid real key requirements
            with patch.dict(os.environ, {f"{provider.upper()}_API_KEY": "test-key"}):
                client = connect_llm(provider=provider, **config)  # type: ignore
                assert client is not None
                assert client.model == config["model"]

        except ImportError:
            # This is the expected outcome if the library is not installed.
            # The test confirms that the factory function works when the library *is* available.
            continue
