import pytest

from agex.llm import LLM, Anthropic, Gemini, OpenAI


def test_llm_dump_config_base():
    """Test dump_config on base LLM."""

    # Define concrete implementation for testing base behavior
    class MockClient(LLM):
        def complete(self, *args, **kwargs):
            pass

        def summarize(self, *args, **kwargs):
            pass

        @property
        def model(self):
            return self._model

        @property
        def provider_name(self):
            return "mock"

    client = MockClient(model="test-model", timeout_seconds=60)
    config = client.dump_config()
    assert config["model"] == "test-model"
    assert config["timeout_seconds"] == 60
    assert config["provider"] == "mock"


def test_llm_from_config_reconstruction():
    """Test reconstruction via from_config."""
    # Note: connect_llm requires API keys usually, so we might need to mock or just check delegation
    # For now, just check basic behavior if we mock connect_llm?
    # Or rely on connect_llm logic.
    pass


@pytest.mark.parametrize(
    "client_cls, provider_name",
    [
        (OpenAI, "openai"),
        (Anthropic, "anthropic"),
        (Gemini, "google"),
    ],
)
def test_provider_dump_config(client_cls, provider_name, monkeypatch):
    """Test that provider clients dump correct config structure."""

    # Mock init to avoid needing real API keys
    monkeypatch.setattr(client_cls, "__init__", lambda self, **kwargs: None)

    # Re-patch attributes manually since init is skipped
    def mock_init(self, model="model", timeout_seconds=90, **kwargs):
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._kwargs = kwargs
        # Add provider-specific attrs
        if client_cls == Gemini:
            self._google_search = kwargs.get("google_search", False)
            self._url_context = kwargs.get("url_context", False)

    monkeypatch.setattr(client_cls, "__init__", mock_init)

    # Create client
    client = client_cls(model="my-model", timeout_seconds=120, temperature=0.7)

    config = client.dump_config()

    assert config["provider"] == provider_name
    assert config["model"] == "my-model"
    assert config["timeout_seconds"] == 120
    assert config["temperature"] == 0.7

    if client_cls == Gemini:
        assert "google_search" in config
        assert "url_context" in config
