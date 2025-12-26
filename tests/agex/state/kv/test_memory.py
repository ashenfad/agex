import pytest

from agex.state.kv import Memory


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


class TestMemoryRemove:
    def test_memory_remove(self):
        store = Memory()
        store.set_many(a=b"1", b=b"2")
        store.remove("a")
        assert "a" not in store
        assert store.get("a") is None
        store.remove_many("b", "missing")
        assert "b" not in store


class TestMemoryCAS:
    """Test compare-and-swap operations for Memory store."""

    def test_memory_cas_success(self):
        store = Memory()
        store.set("key", b"old")

        # CAS should succeed when expected matches
        success = store.cas("key", b"new", expected=b"old")
        assert success is True
        assert store.get("key") == b"new"

    def test_memory_cas_failure(self):
        store = Memory()
        store.set("key", b"value")

        # CAS should fail when expected doesn't match
        success = store.cas("key", b"new", expected=b"wrong")
        assert success is False
        assert store.get("key") == b"value"  # Value unchanged

    def test_memory_cas_create_only(self):
        store = Memory()

        # CAS with expected=None should create if key doesn't exist
        success = store.cas("new_key", b"value", expected=None)
        assert success is True
        assert store.get("new_key") == b"value"

        # CAS with expected=None should fail if key exists
        success = store.cas("new_key", b"updated", expected=None)
        assert success is False
        assert store.get("new_key") == b"value"  # Value unchanged

    def test_memory_cas_type_validation(self):
        store = Memory()

        # CAS should reject non-bytes values
        with pytest.raises(TypeError, match="Expected bytes, got str"):
            store.cas("key", "not bytes", expected=None)  # type: ignore

    def test_memory_cas_thread_safety(self):
        """Test that CAS is atomic even with threading."""
        import threading

        store = Memory()
        store.set("counter", b"0")
        success_count = [0]
        failure_count = [0]

        def increment():
            # Try to increment the counter using CAS
            for _ in range(10):
                current = store.get("counter")
                if current:
                    new_value = str(int(current.decode()) + 1).encode()
                    if store.cas("counter", new_value, expected=current):
                        success_count[0] += 1
                    else:
                        failure_count[0] += 1

        # Run multiple threads trying to increment
        threads = [threading.Thread(target=increment) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Final value should be 50 (5 threads * 10 increments each)
        assert store.get("counter") == b"50"
        assert success_count[0] == 50
