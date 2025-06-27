import shutil
import tempfile

import pytest

from agex.state.kv import Cache, Disk, Memory


class TestMemory:
    """Test the basic Memory KV store."""

    def test_memory_basic_operations(self):
        store = Memory()

        # Test set/get
        store.set("key1", b"value1")
        assert store.get("key1") == b"value1"
        assert store.get("nonexistent") is None

        # Test contains
        assert "key1" in store
        assert "nonexistent" not in store

        # Test items
        store.set("key2", b"value2")
        items = dict(store.items())
        assert items == {"key1": b"value1", "key2": b"value2"}

    def test_memory_get_many_set_many(self):
        store = Memory()

        # Test set_many
        store.set_many(key1=b"value1", key2=b"value2", key3=b"value3")

        # Test get_many
        result = store.get_many("key1", "key3", "nonexistent")
        assert dict(result) == {"key1": b"value1", "key3": b"value3"}


class TestCache:
    """Test the Cache write-through cache."""

    def test_cache_basic_operations(self):
        store = Memory()
        cache = Cache(store, max_bytes=1024)

        # Test set/get (write-through)
        cache.set("key1", b"value1")
        assert cache.get("key1") == b"value1"

        # Verify write-through: data should be in backing store
        assert store.get("key1") == b"value1"

    def test_cache_hit_vs_miss(self):
        store = Memory()
        cache = Cache(store, max_bytes=1024)

        # Put data directly in store (not in cache)
        store.set("key1", b"value1")

        # First get should be a cache miss, but populate cache
        assert cache.get("key1") == b"value1"
        assert "key1" in cache.cache  # Now in cache

        # Second get should be a cache hit
        store.set("key1", b"modified")  # Change backing store
        assert cache.get("key1") == b"value1"  # Still returns cached value

    def test_cache_contains(self):
        store = Memory()
        cache = Cache(store, max_bytes=1024)

        # Test with cached item
        cache.set("cached", b"value")
        assert "cached" in cache

        # Test with store-only item
        store.set("store_only", b"value")
        assert "store_only" in cache

        # Test nonexistent
        assert "nonexistent" not in cache

    def test_cache_get_with_default(self):
        store = Memory()
        cache = Cache(store, max_bytes=1024)

        # Test default behavior
        assert cache.get("nonexistent") is None

        # Ensure default doesn't get cached
        assert "nonexistent" not in cache.cache

    def test_cache_eviction_basic(self):
        store = Memory()
        cache = Cache(store, max_bytes=10)  # Very small cache

        # Add items that exceed max_bytes
        cache.set("key1", b"12345")  # 5 bytes
        cache.set("key2", b"67890")  # 5 bytes, total = 10 bytes (at limit)
        cache.set("key3", b"ABCDE")  # 5 bytes, should trigger eviction

        # key1 should be evicted (FIFO), but still in backing store
        assert "key1" not in cache.cache
        assert cache.get("key1") == b"12345"  # Retrieved from store

        # After getting key1, it's back in cache and may have evicted key2
        # The exact eviction behavior depends on the current cache state
        assert "key1" in cache.cache  # Now cached again
        assert "key3" in cache.cache  # Should still be there

        # Verify all data is still accessible (write-through guarantee)
        assert cache.get("key1") == b"12345"
        assert cache.get("key2") == b"67890"
        assert cache.get("key3") == b"ABCDE"


class TestDisk:
    """Test the Disk KV store."""

    def setup_method(self):
        """Create a temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.store = Disk(self.temp_dir, size_limit=1024 * 1024)  # 1MB for testing

    def teardown_method(self):
        """Clean up temporary directory after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_disk_basic_operations(self):
        # Test set/get
        self.store.set("key1", b"value1")
        assert self.store.get("key1") == b"value1"
        assert self.store.get("nonexistent") is None

        # Test contains
        assert "key1" in self.store
        assert "nonexistent" not in self.store

        # Test items
        self.store.set("key2", b"value2")
        items = dict(self.store.items())
        assert items == {"key1": b"value1", "key2": b"value2"}

    def test_disk_get_many_set_many(self):
        # Test set_many
        self.store.set_many(key1=b"value1", key2=b"value2", key3=b"value3")

        # Test get_many
        result = self.store.get_many("key1", "key3", "nonexistent")
        assert dict(result) == {"key1": b"value1", "key3": b"value3"}

    def test_disk_persistence(self):
        # Store some data
        self.store.set("persistent_key", b"persistent_value")
        assert self.store.get("persistent_key") == b"persistent_value"

        # Create a new Disk instance pointing to the same directory
        new_store = Disk(self.temp_dir)

        # Data should still be there
        assert new_store.get("persistent_key") == b"persistent_value"
        assert "persistent_key" in new_store

        # Items should be accessible
        items = dict(new_store.items())
        assert "persistent_key" in items
        assert items["persistent_key"] == b"persistent_value"

    def test_disk_type_validation(self):
        # Test that non-bytes values are rejected in set()
        with pytest.raises(TypeError, match="Expected bytes, got str"):
            self.store.set("key", "not bytes")  # type: ignore

        with pytest.raises(TypeError, match="Expected bytes, got int"):
            self.store.set("key", 123)  # type: ignore

        # Test that non-bytes values are rejected in set_many()
        with pytest.raises(TypeError, match="Expected bytes for bad_key, got str"):
            self.store.set_many(good_key=b"good", bad_key="bad")  # type: ignore

        # Ensure no partial writes occurred
        assert self.store.get("good_key") is None
        assert self.store.get("bad_key") is None

    def test_disk_clear(self):
        # Add some data
        self.store.set("key1", b"value1")
        self.store.set("key2", b"value2")
        assert "key1" in self.store
        assert "key2" in self.store

        # Clear the store
        self.store.clear()

        # All data should be gone
        assert self.store.get("key1") is None
        assert self.store.get("key2") is None
        assert "key1" not in self.store
        assert "key2" not in self.store
        assert list(self.store.items()) == []

    def test_disk_large_values(self):
        # Test with larger values to ensure proper handling
        large_value = b"x" * 10000  # 10KB
        self.store.set("large_key", large_value)

        retrieved = self.store.get("large_key")
        assert retrieved == large_value
        assert retrieved is not None
        assert len(retrieved) == 10000

    def test_disk_many_keys(self):
        # Test with many keys to ensure proper iteration
        test_data = {f"key_{i}": f"value_{i}".encode() for i in range(100)}

        # Set all keys
        self.store.set_many(**test_data)

        # Verify all keys exist
        for key in test_data:
            assert key in self.store
            assert self.store.get(key) == test_data[key]

        # Verify items() returns everything
        items = dict(self.store.items())
        assert items == test_data

        # Test get_many with subset
        subset_keys = [f"key_{i}" for i in range(0, 100, 10)]  # Every 10th key
        result = self.store.get_many(*subset_keys)
        expected = {k: test_data[k] for k in subset_keys}
        assert dict(result) == expected

    def test_disk_empty_values(self):
        # Test that empty bytes are handled correctly
        self.store.set("empty", b"")
        assert self.store.get("empty") == b""
        assert "empty" in self.store

        # Should appear in items
        items = dict(self.store.items())
        assert items["empty"] == b""
