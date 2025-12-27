"""Tests for ModalDict KV store."""

from unittest.mock import MagicMock, patch

import pytest


class TestModalDict:
    """Test ModalDict KVStore implementation with mocked Modal SDK."""

    @pytest.fixture
    def mock_modal_dict(self):
        """Create a mocked Modal Dict."""
        mock_dict = MagicMock()
        mock_dict.get.return_value = None
        mock_dict.contains.return_value = False
        mock_dict.keys.return_value = iter([])
        mock_dict.items.return_value = iter([])
        mock_dict.put.return_value = True
        return mock_dict

    @pytest.fixture
    def modal_dict_store(self, mock_modal_dict):
        """Create a ModalDict store with mocked backend."""
        with patch("modal.Dict") as MockDict:
            MockDict.from_name.return_value = mock_modal_dict
            from agex.state.kv.modal_dict import ModalDict

            store = ModalDict(name="test-dict", prefix="test-prefix")
            return store, mock_modal_dict

    def test_get_returns_none_for_missing(self, modal_dict_store):
        """get() returns None for missing keys."""
        store, mock = modal_dict_store
        mock.get.return_value = None

        result = store.get("missing")

        assert result is None
        mock.get.assert_called_once_with("test-prefix:missing")

    def test_get_returns_value(self, modal_dict_store):
        """get() returns stored value."""
        store, mock = modal_dict_store
        mock.get.return_value = b"hello"

        result = store.get("key")

        assert result == b"hello"

    def test_set_writes_value(self, modal_dict_store):
        """set() writes value with prefixed key."""
        store, mock = modal_dict_store

        store.set("key", b"value")

        mock.__setitem__.assert_called_once_with("test-prefix:key", b"value")

    def test_set_rejects_non_bytes(self, modal_dict_store):
        """set() raises TypeError for non-bytes."""
        store, _ = modal_dict_store

        with pytest.raises(TypeError, match="Expected bytes"):
            store.set("key", "not bytes")  # type: ignore

    def test_set_many_uses_update(self, modal_dict_store):
        """set_many() uses Modal Dict's update() for batch."""
        store, mock = modal_dict_store

        store.set_many(a=b"1", b=b"2")

        mock.update.assert_called_once_with(
            {"test-prefix:a": b"1", "test-prefix:b": b"2"}
        )

    def test_contains_checks_prefixed_key(self, modal_dict_store):
        """__contains__ checks with prefix."""
        store, mock = modal_dict_store
        mock.contains.return_value = True

        result = "key" in store

        assert result is True
        mock.contains.assert_called_once_with("test-prefix:key")

    def test_remove_uses_pop(self, modal_dict_store):
        """remove() uses pop() to delete key."""
        store, mock = modal_dict_store

        store.remove("key")

        mock.pop.assert_called_once_with("test-prefix:key")

    def test_remove_ignores_missing(self, modal_dict_store):
        """remove() ignores KeyError for missing keys."""
        store, mock = modal_dict_store
        mock.pop.side_effect = KeyError("not found")

        # Should not raise
        store.remove("missing")

    def test_cas_create_uses_skip_if_exists(self, modal_dict_store):
        """CAS with expected=None uses atomic skip_if_exists."""
        store, mock = modal_dict_store
        mock.put.return_value = True

        result = store.cas("key", b"value", expected=None)

        assert result is True
        mock.put.assert_called_once_with(
            "test-prefix:key", b"value", skip_if_exists=True
        )

    def test_cas_create_fails_if_exists(self, modal_dict_store):
        """CAS with expected=None fails if key exists."""
        store, mock = modal_dict_store
        mock.put.return_value = False  # Already exists

        result = store.cas("key", b"value", expected=None)

        assert result is False

    def test_cas_update_checks_current_value(self, modal_dict_store):
        """CAS with expected=bytes checks current value."""
        store, mock = modal_dict_store
        mock.get.return_value = b"old"

        result = store.cas("key", b"new", expected=b"old")

        assert result is True
        mock.get.assert_called_with("test-prefix:key")
        mock.__setitem__.assert_called_once_with("test-prefix:key", b"new")

    def test_cas_update_fails_on_mismatch(self, modal_dict_store):
        """CAS fails if current value doesn't match expected."""
        store, mock = modal_dict_store
        mock.get.return_value = b"different"

        result = store.cas("key", b"new", expected=b"old")

        assert result is False
        mock.__setitem__.assert_not_called()

    def test_keys_filters_by_prefix(self, modal_dict_store):
        """keys() only yields keys matching prefix."""
        store, mock = modal_dict_store
        mock.keys.return_value = iter(
            ["test-prefix:a", "test-prefix:b", "other-prefix:c"]
        )

        keys = list(store.keys())

        assert keys == ["a", "b"]

    def test_items_filters_by_prefix(self, modal_dict_store):
        """items() only yields items matching prefix."""
        store, mock = modal_dict_store
        mock.items.return_value = iter(
            [("test-prefix:a", b"1"), ("test-prefix:b", b"2"), ("other:c", b"3")]
        )

        items = list(store.items())

        assert items == [("a", b"1"), ("b", b"2")]


class TestModalDictNoPrefix:
    """Test ModalDict without prefix."""

    @pytest.fixture
    def store_no_prefix(self):
        """Create ModalDict without prefix."""
        mock_dict = MagicMock()
        mock_dict.get.return_value = None
        with patch("modal.Dict") as MockDict:
            MockDict.from_name.return_value = mock_dict
            from agex.state.kv.modal_dict import ModalDict

            store = ModalDict(name="test-dict", prefix="")
            return store, mock_dict

    def test_no_prefix_uses_raw_keys(self, store_no_prefix):
        """Without prefix, keys are used as-is."""
        store, mock = store_no_prefix

        store.set("key", b"value")

        mock.__setitem__.assert_called_once_with("key", b"value")

    def test_no_prefix_clear_clears_all(self, store_no_prefix):
        """Without prefix, clear() clears entire dict."""
        store, mock = store_no_prefix

        store.clear()

        mock.clear.assert_called_once()
