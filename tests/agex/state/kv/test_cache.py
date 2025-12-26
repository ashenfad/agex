from agex.state.kv import Cache, Memory


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


class TestCacheRemove:
    def test_cache_remove_passthrough(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)
        cache.set_many(a=b"1", b=b"2")

        cache.remove("a")
        assert "a" not in cache
        assert "a" not in backing

        cache.remove_many("b", "missing")
        assert "b" not in cache
        assert "b" not in backing

    def test_cache_remove_clears_cache_entry(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)
        backing.set("x", b"cached")
        # prime cache
        assert cache.get("x") == b"cached"
        assert "x" in cache.cache

        cache.remove("x")
        assert "x" not in cache.cache
        assert "x" not in backing


class TestCacheCAS:
    """Test compare-and-swap operations for Cache store."""

    def test_cache_cas_success_updates_cache(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)

        cache.set("key", b"old")
        assert "key" in cache.cache

        success = cache.cas("key", b"new", expected=b"old")
        assert success is True
        assert cache.get("key") == b"new"
        assert cache.cache["key"] == b"new"  # Cache updated

    def test_cache_cas_failure_invalidates_cache(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)

        cache.set("key", b"value")
        assert "key" in cache.cache

        success = cache.cas("key", b"new", expected=b"wrong")
        assert success is False
        # Cache should be invalidated to prevent stale reads
        assert "key" not in cache.cache

    def test_cache_cas_delegated_to_backing(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)

        # Put data in backing store only (not in cache)
        backing.set("key", b"old")
        assert "key" not in cache.cache

        # CAS should work even though cache doesn't have it
        success = cache.cas("key", b"new", expected=b"old")
        assert success is True
        assert backing.get("key") == b"new"

    def test_cache_cas_create_only(self):
        backing = Memory()
        cache = Cache(backing, max_bytes=1024)

        success = cache.cas("new_key", b"value", expected=None)
        assert success is True
        assert cache.get("new_key") == b"value"
        assert "new_key" in cache.cache
