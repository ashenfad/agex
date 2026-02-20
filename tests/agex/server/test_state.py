"""Tests for server state URI resolution."""

import pytest
from kvit import Staged

from agex.server.state import InvalidStateURIError, resolve_state_uri


class TestResolveStateUri:
    """Tests for resolve_state_uri function."""

    def test_disk_uri_basic(self, tmp_path):
        """Test basic disk:// URI resolution."""
        result = resolve_state_uri("disk://my_session", base_path=str(tmp_path))
        assert isinstance(result, Staged)

    def test_disk_uri_with_underscores(self, tmp_path):
        """Test disk:// URI with underscores in session ID."""
        result = resolve_state_uri("disk://user_123_session", base_path=str(tmp_path))
        assert isinstance(result, Staged)

    def test_disk_uri_with_hyphens(self, tmp_path):
        """Test disk:// URI with hyphens in session ID."""
        result = resolve_state_uri("disk://user-session-456", base_path=str(tmp_path))
        assert isinstance(result, Staged)

    def test_disk_uri_empty_session(self, tmp_path):
        """Test disk:// URI with empty session ID raises error."""
        with pytest.raises(InvalidStateURIError, match="must specify a session ID"):
            resolve_state_uri("disk://", base_path=str(tmp_path))

    def test_disk_uri_path_traversal_blocked(self, tmp_path):
        """Test that path traversal attempts are blocked."""
        # Path traversal in netloc (disk://..) gets rejected by alphanumeric check
        with pytest.raises(InvalidStateURIError, match="alphanumeric"):
            resolve_state_uri("disk://../etc/passwd", base_path=str(tmp_path))

    def test_disk_uri_invalid_characters(self, tmp_path):
        """Test that invalid characters in session ID are rejected."""
        with pytest.raises(InvalidStateURIError, match="alphanumeric"):
            resolve_state_uri("disk://my session", base_path=str(tmp_path))  # space

    def test_unsupported_scheme(self, tmp_path):
        """Test unsupported scheme raises error."""
        with pytest.raises(InvalidStateURIError, match="Unsupported"):
            resolve_state_uri("redis://localhost:6379", base_path=str(tmp_path))

    def test_malformed_uri(self, tmp_path):
        """Test malformed URI raises error."""
        with pytest.raises(InvalidStateURIError):
            resolve_state_uri("not a valid uri :::", base_path=str(tmp_path))
