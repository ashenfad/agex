"""Tests for Modal Volume KV store."""

from unittest.mock import MagicMock, patch

import pytest


class TestModalVolume:
    """Test Volume KVStore implementation with mocked Modal SDK."""

    @pytest.fixture
    def mock_modal_volume(self, tmp_path):
        """Create a mocked Modal Volume that uses local temp directory."""
        mock_volume = MagicMock()
        mock_volume.reload = MagicMock()
        mock_volume.commit = MagicMock()
        return mock_volume

    @pytest.fixture
    def volume_store(self, mock_modal_volume, tmp_path):
        """Create a Volume store with mocked backend."""
        with patch("modal.Volume") as MockVolume:
            MockVolume.from_name.return_value = mock_modal_volume

            from agex.state.kv.modal_volume import Volume

            store = Volume(
                volume_name="test-volume",
                mount_path=str(tmp_path),
                prefix="test-prefix",
            )
            return store, mock_modal_volume, tmp_path

    def test_get_returns_none_for_missing(self, volume_store):
        """get() returns None for missing keys."""
        store, mock, tmp_path = volume_store

        result = store.get("missing")

        assert result is None
        mock.reload.assert_called_once()

    def test_set_writes_file(self, volume_store):
        """set() writes file to correct path."""
        store, mock, tmp_path = volume_store

        store.set("key", b"value")

        expected_path = tmp_path / "test-prefix" / "key"
        assert expected_path.exists()
        assert expected_path.read_bytes() == b"value"

    def test_set_commits_immediately(self, volume_store):
        """set() commits immediately after each write."""
        store, mock, tmp_path = volume_store

        store.set("key", b"value")

        mock.commit.assert_called_once()

    def test_get_returns_written_value(self, volume_store):
        """get() returns value after set()."""
        store, mock, tmp_path = volume_store

        store.set("key", b"value")
        result = store.get("key")

        assert result == b"value"

    def test_set_rejects_non_bytes(self, volume_store):
        """set() raises TypeError for non-bytes."""
        store, _, _ = volume_store

        with pytest.raises(TypeError, match="Expected bytes"):
            store.set("key", "not bytes")  # type: ignore

    def test_set_many_single_commit(self, volume_store):
        """set_many() batches writes with single commit."""
        store, mock, tmp_path = volume_store

        store.set_many(a=b"1", b=b"2", c=b"3")

        # Should have one commit for all three writes
        mock.commit.assert_called_once()

        # All files should exist
        assert (tmp_path / "test-prefix" / "a").read_bytes() == b"1"
        assert (tmp_path / "test-prefix" / "b").read_bytes() == b"2"
        assert (tmp_path / "test-prefix" / "c").read_bytes() == b"3"

    def test_keys_lists_files(self, volume_store):
        """keys() yields all stored keys."""
        store, mock, tmp_path = volume_store

        store.set_many(a=b"1", b=b"2")

        keys = set(store.keys())

        assert keys == {"a", "b"}

    def test_items_yields_pairs(self, volume_store):
        """items() yields key-value pairs."""
        store, mock, tmp_path = volume_store

        store.set_many(a=b"1", b=b"2")

        items = dict(store.items())

        assert items == {"a": b"1", "b": b"2"}

    def test_remove_deletes_file(self, volume_store):
        """remove() deletes the file."""
        store, mock, tmp_path = volume_store

        store.set("key", b"value")
        mock.commit.reset_mock()

        store.remove("key")

        assert "key" not in store
        mock.commit.assert_called_once()

    def test_remove_missing_no_error(self, volume_store):
        """remove() doesn't error for missing keys."""
        store, mock, tmp_path = volume_store

        # Should not raise, and should not commit
        store.remove("missing")
        mock.commit.assert_not_called()

    def test_contains_checks_existence(self, volume_store):
        """__contains__ checks file existence."""
        store, mock, tmp_path = volume_store

        assert "key" not in store

        store.set("key", b"value")

        assert "key" in store


class TestModalVolumeCAS:
    """Test CAS behavior with Modal Volume."""

    @pytest.fixture
    def volume_store(self, tmp_path):
        """Create a Volume store with mocked backend."""
        mock_volume = MagicMock()

        with patch("modal.Volume") as MockVolume:
            MockVolume.from_name.return_value = mock_volume

            from agex.state.kv.modal_volume import Volume

            store = Volume(
                volume_name="test-volume",
                mount_path=str(tmp_path),
                prefix="test",
            )
            return store, mock_volume, tmp_path

    def test_cas_create_succeeds_when_missing(self, volume_store):
        """CAS with expected=None succeeds when key doesn't exist."""
        store, mock, tmp_path = volume_store

        result = store.cas("key", b"value", expected=None)

        assert result is True
        assert store.get("key") == b"value"

    def test_cas_create_fails_when_exists(self, volume_store):
        """CAS with expected=None fails when key exists."""
        store, mock, tmp_path = volume_store

        store.set("key", b"existing")

        result = store.cas("key", b"new", expected=None)

        assert result is False
        assert store.get("key") == b"existing"

    def test_cas_update_succeeds_on_match(self, volume_store):
        """CAS with expected=bytes succeeds when values match."""
        store, mock, tmp_path = volume_store

        store.set("key", b"old")

        result = store.cas("key", b"new", expected=b"old")

        assert result is True
        assert store.get("key") == b"new"

    def test_cas_update_fails_on_mismatch(self, volume_store):
        """CAS fails when current value doesn't match expected."""
        store, mock, tmp_path = volume_store

        store.set("key", b"actual")

        result = store.cas("key", b"new", expected=b"wrong")

        assert result is False
        assert store.get("key") == b"actual"


class TestModalVolumeKeyEncoding:
    """Test key encoding for filesystem safety."""

    @pytest.fixture
    def volume_store(self, tmp_path):
        """Create a Volume store with mocked backend."""
        mock_volume = MagicMock()

        with patch("modal.Volume") as MockVolume:
            MockVolume.from_name.return_value = mock_volume

            from agex.state.kv.modal_volume import Volume

            store = Volume(
                volume_name="test-volume",
                mount_path=str(tmp_path),
                prefix="",
            )
            return store, tmp_path

    def test_slashes_encoded(self, volume_store):
        """Keys with slashes are encoded properly."""
        store, tmp_path = volume_store

        store.set("path/to/key", b"value")

        # Should be stored as path__to__key, not in subdirectory
        assert (tmp_path / "path__to__key").exists()
        assert not (tmp_path / "path").exists()

    def test_encoded_key_retrievable(self, volume_store):
        """Keys with slashes can be retrieved."""
        store, tmp_path = volume_store

        store.set("a/b/c", b"value")
        result = store.get("a/b/c")

        assert result == b"value"

    def test_keys_decodes_properly(self, volume_store):
        """keys() returns decoded keys."""
        store, tmp_path = volume_store

        store.set("path/to/key", b"value")

        keys = list(store.keys())

        assert "path/to/key" in keys
