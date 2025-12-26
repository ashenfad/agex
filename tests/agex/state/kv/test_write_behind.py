from agex.state.kv import Memory, WriteBehind


class TestWriteBehind:
    """Test the WriteBehind KV store wrapper."""

    def test_write_behind_basic_operations(self):
        store = Memory()
        wb = WriteBehind(store)

        # Test set (async)
        wb.set("key1", b"value1")

        # Should be eventually consistent (auto-flush on get)
        assert store.get("key1") is None  # Store might not have it yet (async)
        assert wb.get("key1") == b"value1"  # Wrapper should have it (auto-flush)

        # Test contains
        assert "key1" in wb
        assert "nonexistent" not in wb

        # Test items
        wb.set("key2", b"value2")
        # items() should auto-flush now
        items = dict(wb.items())
        assert items == {"key1": b"value1", "key2": b"value2"}

    def test_write_behind_set_many(self):
        store = Memory()
        wb = WriteBehind(store)

        # Test set_many (async)
        wb.set_many(key1=b"value1", key2=b"value2")

        # Should auto-flush on get_many
        result = wb.get_many("key1", "key2", "nonexistent")
        assert dict(result) == {"key1": b"value1", "key2": b"value2"}

    def test_write_behind_error_handling(self, capsys):
        """Test that background errors don't crash the main thread."""
        store = Memory()
        wb = WriteBehind(store)

        # Force an error by passing invalid type to underlying store
        # Memory.set raises TypeError for non-bytes
        wb.set("key", "not bytes")  # type: ignore

        wb.flush()

        # Main thread should continue fine
        wb.set("key2", b"valid")
        wb.flush()
        assert wb.get("key2") == b"valid"

        # Check stderr for the error message
        captured = capsys.readouterr()
        assert "WriteBehind error" in captured.err
        assert "Expected bytes" in captured.err

    def test_write_behind_remove(self):
        store = Memory()
        wb = WriteBehind(store)
        wb.set_many(a=b"1", b=b"2")
        wb.flush()

        wb.remove("a")
        wb.flush()
        assert "a" not in wb
        assert store.get("a") is None

        wb.remove_many("b", "missing")
        wb.flush()
        assert "b" not in wb
        assert store.get("b") is None


class TestWriteBehindCAS:
    """Test compare-and-swap operations for WriteBehind store."""

    def test_write_behind_cas_flushes_first(self):
        backing = Memory()
        wb = WriteBehind(backing)

        # Queue a write
        wb.set("key", b"old")

        # CAS should flush pending writes first
        success = wb.cas("key", b"new", expected=b"old")
        assert success is True

        # Verify in backing store (proves flush happened)
        assert backing.get("key") == b"new"

    def test_write_behind_cas_failure(self):
        backing = Memory()
        wb = WriteBehind(backing)

        wb.set("key", b"value")
        wb.flush()

        success = wb.cas("key", b"new", expected=b"wrong")
        assert success is False
        assert wb.get("key") == b"value"

    def test_write_behind_cas_with_pending_writes(self):
        backing = Memory()
        wb = WriteBehind(backing)

        # Queue multiple writes
        wb.set("key1", b"value1")
        wb.set("key2", b"value2")
        wb.set("key3", b"value3")

        # CAS should flush everything
        success = wb.cas("key2", b"updated", expected=b"value2")
        assert success is True

        # All queued writes should be in backing store
        assert backing.get("key1") == b"value1"
        assert backing.get("key2") == b"updated"
        assert backing.get("key3") == b"value3"

    def test_write_behind_cas_create_only(self):
        backing = Memory()
        wb = WriteBehind(backing)

        success = wb.cas("new_key", b"value", expected=None)
        assert success is True
