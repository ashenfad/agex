"""Tests for Composite N-tier caching."""

import pytest

from agex.state.kv.composite import Composite
from agex.state.kv.memory import Memory


class TestCompositeBasic:
    """Test basic Composite functionality."""

    def test_requires_at_least_one_store(self):
        """Composite requires at least one store."""
        with pytest.raises(ValueError, match="at least one store"):
            Composite([])

    def test_single_tier_works(self):
        """Single-tier cache acts as pass-through."""
        store = Memory()
        cache = Composite([store])

        cache.set("key", b"value")
        assert cache.get("key") == b"value"

    def test_two_tier_cache_hit(self):
        """Two-tier: L1 hit doesn't touch L2."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Write goes to both
        cache.set("key", b"value")
        assert l1.get("key") == b"value"
        assert l2.get("key") == b"value"

        # Read from L1 (fast path)
        assert cache.get("key") == b"value"

    def test_two_tier_cache_miss_populates_l1(self):
        """Two-tier: L2 hit populates L1."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Manually write to L2 only
        l2.set("key", b"value")

        # First read misses L1, hits L2, populates L1
        assert cache.get("key") == b"value"
        assert l1.get("key") == b"value"  # Now cached in L1

    def test_three_tier_cascade(self):
        """Three-tier: reads cascade through all tiers."""
        l1 = Memory()
        l2 = Memory()
        l3 = Memory()
        cache = Composite([l1, l2, l3])

        # Manually write to L3 only
        l3.set("key", b"value")

        # First read: miss L1, miss L2, hit L3
        # Should populate L1 and L2
        assert cache.get("key") == b"value"
        assert l1.get("key") == b"value"
        assert l2.get("key") == b"value"

    def test_write_to_all_tiers(self):
        """Writes propagate to all tiers."""
        l1 = Memory()
        l2 = Memory()
        l3 = Memory()
        cache = Composite([l1, l2, l3])

        cache.set("key", b"value")

        # All tiers should have the value
        assert l1.get("key") == b"value"
        assert l2.get("key") == b"value"
        assert l3.get("key") == b"value"

    def test_set_many_to_all_tiers(self):
        """set_many propagates to all tiers."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        cache.set_many(a=b"1", b=b"2")

        assert l1.get("a") == b"1"
        assert l2.get("a") == b"1"
        assert l1.get("b") == b"2"
        assert l2.get("b") == b"2"


class TestCompositeCAS:
    """Test CAS behavior with Composite."""

    def test_cas_delegates_to_last_tier(self):
        """CAS operates on the most durable tier."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Create-if-missing on L2
        success = cache.cas("key", b"new", expected=None)
        assert success
        assert l2.get("key") == b"new"
        assert l1.get("key") == b"new"  # Populated on success

    def test_cas_update_populates_caches(self):
        """CAS update populates all cache tiers."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Initial write
        cache.set("key", b"old")

        # CAS update
        success = cache.cas("key", b"new", expected=b"old")
        assert success
        assert l1.get("key") == b"new"
        assert l2.get("key") == b"new"


class TestCompositeStats:
    """Test hit/miss tracking."""

    def test_hit_tracking(self):
        """Hits are tracked correctly."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Write and read from L1 (no hit counted on write)
        cache.set("key", b"value")
        cache.get("key")  # L1 hit (not counted)

        # Remove from L1, read from L2
        l1.remove("key")
        cache.get("key")  # L2 hit (counted)
        assert cache._hits == 1

    def test_miss_tracking(self):
        """Misses are tracked correctly."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Read non-existent key
        assert cache.get("missing") is None
        assert cache._misses == 1


class TestCompositeKeyIteration:
    """Test keys/items delegation to authoritative tier."""

    def test_keys_from_last_tier(self):
        """keys() delegates to most durable tier."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        # Only L2 is authoritative
        l2.set("key1", b"1")
        l2.set("key2", b"2")

        keys = set(cache.keys())
        assert keys == {"key1", "key2"}

    def test_items_from_last_tier(self):
        """items() delegates to most durable tier."""
        l1 = Memory()
        l2 = Memory()
        cache = Composite([l1, l2])

        l2.set("key1", b"1")
        l2.set("key2", b"2")

        items = dict(cache.items())
        assert items == {"key1": b"1", "key2": b"2"}
