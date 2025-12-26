import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from agex.state.kv import File


class TestFile:
    """Test the File store."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.store = File(self.temp_dir)  # Defaults to atomic_writes=True

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_file_basic_operations(self):
        # Test set/get
        self.store.set("key1", b"value1")
        assert self.store.get("key1") == b"value1"
        assert self.store.get("nonexistent") is None

        # Test contains
        assert "key1" in self.store
        assert "nonexistent" not in self.store

    def test_file_persistence(self):
        self.store.set("key", b"value")
        # New instance pointing to same dir
        new_store = File(self.temp_dir)
        assert new_store.get("key") == b"value"

    def test_file_items_keys(self):
        # Use specialized keys to test base64 encoding
        data = {
            "key1": b"val1",
            "key/with/slash": b"val2",
            "key_with_underscores": b"val3",
            "key.with.dots": b"val4",
            "💩": b"unicode",
        }
        self.store.set_many(**data)

        # Verify keys()
        keys = list(self.store.keys())
        assert len(keys) == 5
        for k in data:
            assert k in keys

        # Verify items()
        items = dict(self.store.items())
        assert items == data

    def test_file_remove(self):
        self.store.set("key", b"val")
        self.store.remove("key")
        assert self.store.get("key") is None
        assert "key" not in self.store

        # Remove nonexistent should be safe
        self.store.remove("nonexistent")

    def test_file_cas(self):
        self.store.set("key", b"old")

        # Success
        assert self.store.cas("key", b"new", expected=b"old")
        assert self.store.get("key") == b"new"

        # Failure
        assert not self.store.cas("key", b"newer", expected=b"wrong")
        assert self.store.get("key") == b"new"

        # Create only
        assert self.store.cas("created", b"val", expected=None)
        assert self.store.get("created") == b"val"

    def test_file_atomic_config(self):
        """Verify atomic_writes parameter is accepted."""
        store_atomic = File(self.temp_dir, atomic_writes=True)
        assert store_atomic.atomic_writes is True

        store_direct = File(self.temp_dir, atomic_writes=False)
        assert store_direct.atomic_writes is False


class TestFileBehavior:
    """Test specific behavior of atomic vs direct writes."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_atomic_writes_create_temp_file(self):
        """Verify that atomic=True creates a temp file."""
        # It's hard to catch the temp file in existence during a fast test
        # without mocking. So let's mock the write to fail and ensure
        # no partial file is left behind.

        store = File(self.temp_dir, atomic_writes=True)

        # Mock write_bytes to fail
        with pytest.raises(RuntimeError):
            with mock.patch(
                "pathlib.Path.write_bytes", side_effect=RuntimeError("Fail")
            ):
                store.set("key", b"value")

        # With atomic writes, the temp file should be cleaned up
        # and the target file should not exist
        assert store.get("key") is None
        # Check no .tmp files left
        assert len(list(Path(self.temp_dir).glob("*.tmp"))) == 0

    def test_direct_writes_leave_partial_on_failure(self):
        """Verify that atomic=False writes directly."""
        store = File(self.temp_dir, atomic_writes=False)

        # Even if write fails midway (simulated), bytes were attempted
        # directly on the path.
        # We can't easily simulate "half-written" in python without lower level mocks.
        # Instead, let's verify that NO temp file is used by spying on Path.rename.

        with mock.patch("pathlib.Path.rename") as mock_rename:
            store.set("key", b"value")

            # Rename should NOT be called for atomic_writes=False
            mock_rename.assert_not_called()

        assert store.get("key") == b"value"

    def test_cas_always_atomic(self):
        """Verify CAS is always atomic regardless of config."""

        # Case 1: Atomic=True (default) -> Uses Rename
        store_atomic = File(self.temp_dir, atomic_writes=True)
        store_atomic.set("key", b"old")

        with mock.patch("pathlib.Path.rename") as mock_rename:
            store_atomic.cas("key", b"new", expected=b"old")
            mock_rename.assert_called()  # Should use rename

        # Case 2: Atomic=False -> STILL Uses Rename
        store_direct = File(self.temp_dir, atomic_writes=False)
        store_direct.set("key", b"old")

        with mock.patch("pathlib.Path.rename") as mock_rename:
            store_direct.cas("key", b"new", expected=b"old")
            mock_rename.assert_called()  # Should use rename (for safety)


class TestFileConcurrency:
    """Test parallel/concurrent features of File."""

    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_parallelism_config(self):
        """Verify parallelism parameter creates executor."""
        # Default = 10
        store = File(self.temp_dir)
        assert store.parallelism == 10
        assert store._executor is not None
        assert store._executor._max_workers == 10

        # Custom
        store = File(self.temp_dir, parallelism=4)
        assert store.parallelism == 4
        assert store._executor._max_workers == 4

        # Disabled
        store = File(self.temp_dir, parallelism=1)
        assert store.parallelism == 1
        assert store._executor is None

    def test_set_many_parallel(self):
        """Verify set_many works with parallelism enabled."""
        store = File(self.temp_dir, parallelism=4)
        data = {f"k{i}": f"v{i}".encode() for i in range(20)}

        store.set_many(**data)

        # Verify persistence
        stored = store.get_many(*data.keys())
        assert len(stored) == 20
        assert stored == data

    def test_get_many_parallel(self):
        """Verify get_many works with parallelism enabled."""
        store = File(self.temp_dir, parallelism=4)
        data = {f"k{i}": f"v{i}".encode() for i in range(20)}
        store.set_many(**data)

        keys = list(data.keys())
        # Add some missing keys
        query_keys = keys + ["missing1", "missing2"]

        result = store.get_many(*query_keys)
        assert len(result) == 20
        assert result == data

    def test_remove_many_parallel(self):
        """Verify remove_many works with parallelism enabled."""
        store = File(self.temp_dir, parallelism=4)
        data = {f"k{i}": f"v{i}".encode() for i in range(20)}
        store.set_many(**data)

        keys_to_remove = [f"k{i}" for i in range(10)]
        store.remove_many(*keys_to_remove)

        # Verify first 10 gone
        for k in keys_to_remove:
            assert store.get(k) is None

        # Verify remaining 10 exist
        for i in range(10, 20):
            assert store.get(f"k{i}") is not None

    def test_parallel_error_propagation(self):
        """Verify exceptions in worker threads propagate to caller."""
        store = File(self.temp_dir, parallelism=2)

        # Mock set to raise exception for a specific key
        original_set = store.set

        def mock_set(key, value):
            if key == "fail":
                raise RuntimeError("Failed write")
            original_set(key, value)

        store.set = mock_set

        with pytest.raises(RuntimeError, match="Failed write"):
            store.set_many(ok1=b"1", fail=b"2", ok2=b"3")
