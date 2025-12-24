"""Tests for connect_host factory function."""

import pytest

from agex.host import HTTP, Local, connect_host


class TestConnectHost:
    """Tests for the connect_host factory."""

    def test_default_is_local(self):
        """Test that default provider is local."""
        host = connect_host()
        assert isinstance(host, Local)

    def test_explicit_local(self):
        """Test explicit local provider."""
        host = connect_host(provider="local")
        assert isinstance(host, Local)

    def test_http_requires_url(self):
        """Test that HTTP provider requires URL."""
        with pytest.raises(ValueError, match="requires 'url'"):
            connect_host(provider="http")

    def test_http_with_url(self):
        """Test HTTP provider with URL."""
        host = connect_host(provider="http", url="http://localhost:8000")
        assert isinstance(host, HTTP)
        assert host.url == "http://localhost:8000"

    def test_http_with_options(self):
        """Test HTTP provider with additional options."""
        host = connect_host(
            provider="http",
            url="http://localhost:8000",
            timeout=60.0,
            retries=3,
        )
        assert isinstance(host, HTTP)
        assert host.timeout == 60.0
        assert host.retries == 3

    def test_unknown_provider_rejected(self):
        """Test that unknown providers are rejected."""
        with pytest.raises(ValueError, match="Unknown host provider"):
            connect_host(provider="unknown")
