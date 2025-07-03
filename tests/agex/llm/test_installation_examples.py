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
        client = get_llm_client(provider="openai")
        # If we get here, OpenAI is installed and working
        assert client.model == "gpt-4.1-nano"  # Default model
        print("✅ OpenAI provider is available")
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
        client = get_llm_client(provider="anthropic")
        # If we get here, Anthropic is installed and working
        assert client.model == "gpt-4.1-nano"  # Uses global default model
        print("✅ Anthropic provider is available")
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
        client = get_llm_client(provider="gemini")
        # If we get here, Gemini is installed and working
        assert client.model == "gpt-4.1-nano"  # Uses global default model
        print("✅ Gemini provider is available")
    except ImportError as e:
        # This would happen if google-generativeai package isn't installed
        assert "pip install 'agex[gemini]'" in str(e)
        print("❌ Gemini provider requires installation")


def test_all_providers_installation():
    """
    Test that all providers can be installed together.

    This simulates: pip install "agex[all-providers]"
    """
    providers_to_test = ["dummy", "openai", "anthropic", "gemini"]
    available_providers = []

    for provider in providers_to_test:
        try:
            client = get_llm_client(provider=provider)  # type: ignore
            available_providers.append(provider)
            print(f"✅ {provider} provider is available")
        except ImportError:
            print(f"❌ {provider} provider requires installation")

    # Dummy should always be available
    assert "dummy" in available_providers

    # Print summary
    print(f"\nAvailable providers: {available_providers}")
    print(f"Total providers available: {len(available_providers)}/4")


# Note: This file can be run as a pytest module but not directly due to import path issues
# Use: python -m pytest tests/agex/llm/test_installation_examples.py -v
