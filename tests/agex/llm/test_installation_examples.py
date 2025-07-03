"""
Installation examples and verification tests.

This file documents the various ways users can install agex and verifies
that the installation instructions in the docs work correctly.
"""

from agex.llm import get_llm_client


def test_basic_installation_works():
    """
    Test that basic installation (core + dummy client) works.

    This simulates: pip install agex
    """
    # Dummy client should always be available
    client = get_llm_client(provider="dummy")
    assert client.model == "dummy"


def test_openai_provider_installation():
    """
    Test OpenAI provider installation.

    This simulates: pip install "agex[openai]"
    """
    try:
        # Set a dummy API key for testing client creation
        import os

        original_key = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "test-key-for-installation-test"

        try:
            # Explicitly pass configuration to avoid environment variable interference
            client = get_llm_client(provider="openai", model="gpt-4.1-nano")
            # If we get here, OpenAI is installed and working
            assert client.model == "gpt-4.1-nano"
            assert client.provider_name == "OpenAI"
            print("✅ OpenAI provider is available")
        finally:
            # Restore original API key state
            if original_key is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = original_key

    except ImportError as e:
        # This would happen if openai package isn't installed
        assert "pip install 'agex[openai]'" in str(e)
        print("❌ OpenAI provider requires installation")


def test_anthropic_provider_installation():
    """
    Test Anthropic provider installation.

    This simulates: pip install "agex[anthropic]"
    """
    try:
        # Set a dummy API key for testing client creation
        import os

        original_key = os.environ.get("ANTHROPIC_API_KEY")
        os.environ["ANTHROPIC_API_KEY"] = "test-key-for-installation-test"

        try:
            # Explicitly pass configuration to avoid environment variable interference
            client = get_llm_client(
                provider="anthropic", model="claude-3-sonnet-20240229"
            )
            # If we get here, Anthropic is installed and working
            assert client.model == "claude-3-sonnet-20240229"
            assert client.provider_name == "Anthropic"
            print("✅ Anthropic provider is available")
        finally:
            # Restore original API key state
            if original_key is None:
                os.environ.pop("ANTHROPIC_API_KEY", None)
            else:
                os.environ["ANTHROPIC_API_KEY"] = original_key

    except ImportError as e:
        # This would happen if anthropic package isn't installed
        assert "pip install 'agex[anthropic]'" in str(e)
        print("❌ Anthropic provider requires installation")


def test_gemini_provider_installation():
    """
    Test Gemini provider installation.

    This simulates: pip install "agex[gemini]"
    """
    try:
        # Set a dummy API key for testing client creation
        import os

        original_key = os.environ.get("GEMINI_API_KEY")
        os.environ["GEMINI_API_KEY"] = "test-key-for-installation-test"

        try:
            # Explicitly pass configuration to avoid environment variable interference
            client = get_llm_client(provider="gemini", model="gemini-1.5-flash")
            # If we get here, Gemini is installed and working
            assert client.model == "gemini-1.5-flash"
            assert client.provider_name == "Google Gemini"
            print("✅ Gemini provider is available")
        finally:
            # Restore original API key state
            if original_key is None:
                os.environ.pop("GEMINI_API_KEY", None)
            else:
                os.environ["GEMINI_API_KEY"] = original_key

    except ImportError as e:
        # This would happen if google-generativeai package isn't installed
        assert "pip install 'agex[gemini]'" in str(e)
        print("❌ Gemini provider requires installation")


def test_all_providers_installation():
    """
    Test that all providers can be installed together.

    This simulates: pip install "agex[all-providers]"
    """
    import os

    # Set dummy API keys for testing client creation
    api_keys_to_set = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GEMINI_API_KEY": "test-gemini-key",
    }

    # Store original keys
    original_keys = {}
    for key in api_keys_to_set:
        original_keys[key] = os.environ.get(key)
        os.environ[key] = api_keys_to_set[key]

    try:
        # Test each provider with explicit configuration to avoid environment variable interference
        test_configs = [
            ("dummy", {}),
            ("openai", {"model": "gpt-4.1-nano"}),
            ("anthropic", {"model": "claude-3-sonnet-20240229"}),
            ("gemini", {"model": "gemini-1.5-flash"}),
        ]
        available_providers = []

        for provider, config in test_configs:
            try:
                client = get_llm_client(provider=provider, **config)  # type: ignore
                available_providers.append(provider)
                print(f"✅ {provider} provider is available (model: {client.model})")
            except ImportError:
                print(f"❌ {provider} provider requires installation")

        # Dummy should always be available
        assert "dummy" in available_providers

        # Print summary
        print(f"\nAvailable providers: {available_providers}")
        print(f"Total providers available: {len(available_providers)}/4")

    finally:
        # Restore original API key state
        for key, original_value in original_keys.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


# Note: This file can be run as a pytest module but not directly due to import path issues
# Use: python -m pytest tests/agex/llm/test_installation_examples.py -v
