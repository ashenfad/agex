import pickle
import time
from dataclasses import dataclass

from agex.state.versioned import (
    COMMIT_KEYSET,
    HEAD_COMMIT,
    META_KEY,
    PARENT_COMMIT,
    TOTAL_VAR_SIZE_KEY,
    MetaEntry,
    SnapshotResult,
    Versioned,
    get_commit_hash,
)


def _is_system_key(key: str) -> bool:
    """
    Check if a key is a system key (starts with __).

    Handles both direct keys ("__event_log__") and namespaced keys
    ("agent_name/__event_log__") by extracting the base key name.
    """
    # Extract base key name after any namespace prefix
    # e.g., "agent_name/__event_log__" -> "__event_log__"
    base_key = key.split("/")[-1] if "/" in key else key
    return base_key.startswith("__")


@dataclass(frozen=True)
class RebaseResult:
    performed: bool
    new_commit: str | None
    dropped_keys: tuple[str, ...]
    kept_keys: tuple[str, ...]
    total_size_before: int
    total_size_after: int
    orphans_cleaned: int = 0


class GCVersioned(Versioned):
    """
    Versioned state with built-in garbage collection via a "rebase" strategy.

    Rebase strategy (high/low water):
    - Track total persisted user-var size from Versioned metadata.
    - If total <= high_water_bytes: no-op.
    - If total > high_water_bytes: drop coldest user keys (oldest touch, then largest)
      until total <= low_water_bytes (default 80% of high_water_bytes). System keys
      (`__*` such as event logs/meta) are always retained.
    - Write a fresh head commit containing only retained keys and updated metadata,
      then delete dropped blobs.

    This keeps the state bounded without changing user code; every snapshot auto-runs
    the high/low check.
    """

    def __init__(
        self,
        store=None,
        commit_hash: str | None = None,
        *,
        high_water_bytes: int,
        low_water_bytes: int | None = None,
    ):
        """
        Args:
            store: KVStore backend (Memory, Disk, Cache, etc.).
            commit_hash: Optional head commit to start from.
            high_water_bytes: Trigger threshold; rebasing runs when total user-var
                size exceeds this.
            low_water_bytes: Target threshold; rebasing prunes until total user-var
                size is at or below this. Defaults to 80% of high_water_bytes.
        """
        super().__init__(store, commit_hash=commit_hash)
        if high_water_bytes <= 0:
            raise ValueError("high_water_bytes must be > 0")
        self.high_water = high_water_bytes
        self.low_water = (
            low_water_bytes
            if low_water_bytes is not None
            else int(high_water_bytes * 0.8)
        )
        if self.low_water <= 0 or self.low_water > self.high_water:
            self.low_water = int(high_water_bytes * 0.8)
        self.last_rebase_result: RebaseResult | None = None

    def maybe_rebase(self) -> RebaseResult:
        total = self._load_total_size()
        if total <= self.high_water:
            return RebaseResult(
                performed=False,
                new_commit=None,
                dropped_keys=(),
                kept_keys=tuple(self.commit_keys.keys()),
                total_size_before=total,
                total_size_after=total,
            )
        return self.rebase()

    def rebase(self) -> RebaseResult:
        current_commit = self.current_commit
        if current_commit is None:
            return RebaseResult(
                performed=False,
                new_commit=None,
                dropped_keys=(),
                kept_keys=(),
                total_size_before=0,
                total_size_after=0,
            )

        meta = self._load_meta(current_commit)
        total_before = self._load_total_size(
            default=sum(entry.size or 0 for entry in meta.values())
        )

        # Identify system and user keys
        system_keys = {k: v for k, v in self.commit_keys.items() if _is_system_key(k)}
        user_meta = {k: v for k, v in meta.items() if not _is_system_key(k)}

        # Resolve event references and drop unreferenced event blobs preemptively.
        event_refs = set(self.get("__event_log__", []))
        event_keys = {k for k in self.commit_keys if k.startswith("_event_")}
        unref_events = event_keys - event_refs

        retained_keys = set(system_keys.keys()) | set(user_meta.keys())
        total = sum(entry.size or 0 for entry in user_meta.values())
        dropped: list[str] = []

        for key in unref_events:
            if key in retained_keys:
                retained_keys.discard(key)
            if key in user_meta:
                size = user_meta[key].size or 0
                total -= size
                user_meta.pop(key, None)
            dropped.append(key)

        # Always retain referenced events (they should not be dropped by GC).
        retained_keys |= event_refs

        # Calculate drop list ordered by oldest touch then largest size
        # Exclude referenced events from candidates to prevent them from being dropped
        candidates: list[tuple[str, MetaEntry]] = sorted(
            ((k, v) for k, v in user_meta.items() if k not in event_refs),
            key=lambda kv: (kv[1].last_touch, -(kv[1].size or 0)),
        )

        for key, entry in candidates:
            if total <= self.low_water:
                break
            retained_keys.discard(key)
            dropped.append(key)
            total -= entry.size or 0

        # Build new commit
        new_hash = get_commit_hash()
        new_commit_keys: dict[str, str] = {}
        new_meta: dict[str, MetaEntry] = {}
        diffs: dict[str, bytes] = {}

        # Carry system keys as-is
        for key, versioned_key in system_keys.items():
            serialized = self.long_term.get(versioned_key)
            if serialized is None:
                continue
            new_commit_keys[key] = (
                f"{new_hash}:{key}" if ":" not in versioned_key else versioned_key
            )
            diffs[new_commit_keys[key]] = serialized

        # Carry retained user keys
        for key in retained_keys:
            if _is_system_key(key):
                continue
            versioned_key = self.commit_keys.get(key)
            if not versioned_key:
                continue
            serialized = self.long_term.get(versioned_key)
            if serialized is None:
                continue
            new_vk = f"{new_hash}:{key}"
            new_commit_keys[key] = new_vk
            diffs[new_vk] = serialized
            if key in meta:
                new_meta[key] = meta[key]

        # Diff keys for this commit: user keys we kept
        diffs["__diff_keys__"] = pickle.dumps(
            tuple(k for k in retained_keys if not _is_system_key(k))
        )

        diffs[COMMIT_KEYSET % new_hash] = pickle.dumps(new_commit_keys)
        diffs[PARENT_COMMIT % new_hash] = pickle.dumps(None)
        diffs[HEAD_COMMIT] = pickle.dumps(new_hash)
        diffs[META_KEY % new_hash] = pickle.dumps(new_meta)
        total_after = sum(entry.size or 0 for entry in new_meta.values())
        diffs[TOTAL_VAR_SIZE_KEY % new_hash] = pickle.dumps(total_after)

        self.long_term.set_many(**diffs)

        # Delete dropped keys' blobs
        to_delete = []
        for key in dropped:
            vk = self.commit_keys.get(key)
            if vk:
                to_delete.append(vk)
        if to_delete:
            self.long_term.remove_many(*to_delete)

        # Update in-memory state
        self.commit_keys = new_commit_keys
        self.current_commit = new_hash
        self.removed = set()
        self.live = self.live.__class__()  # reset live
        self.accessed_objects.clear()
        self.meta = new_meta

        # Clean orphaned commits (from failed CAS attempts)
        orphans_cleaned = self.clean_orphans(min_age_seconds=3600)

        return RebaseResult(
            performed=True,
            new_commit=new_hash,
            dropped_keys=tuple(dropped),
            kept_keys=tuple(retained_keys),
            total_size_before=total_before,
            total_size_after=total_after,
            orphans_cleaned=orphans_cleaned,
        )

    def snapshot(self) -> SnapshotResult:
        result = super().snapshot()

        rebase_result = self.maybe_rebase()
        self.last_rebase_result = rebase_result
        # If rebase produced a new head, reflect that in the snapshot result.
        if rebase_result.performed and rebase_result.new_commit:
            result.commit_hash = rebase_result.new_commit

        return result

    def _load_meta(self, commit_hash: str) -> dict[str, MetaEntry]:
        meta_bytes = self.long_term.get(META_KEY % commit_hash)
        if meta_bytes is None:
            return {}
        try:
            loaded_meta = pickle.loads(meta_bytes)
            # Handle legacy 2-tuple format by converting to MetaEntry
            if loaded_meta and isinstance(next(iter(loaded_meta.values())), tuple):
                first_entry = next(iter(loaded_meta.values()))
                if len(first_entry) == 2:
                    # Legacy format: convert to MetaEntry
                    return {
                        k: MetaEntry(touch, size, time.time())
                        for k, (touch, size) in loaded_meta.items()
                    }
            # Already in MetaEntry format or new 3-tuple format
            if loaded_meta and isinstance(next(iter(loaded_meta.values())), tuple):
                # Convert 3-tuple to MetaEntry
                result = {}
                for k, v in loaded_meta.items():
                    if isinstance(v, MetaEntry):
                        result[k] = v
                    elif len(v) == 3:
                        result[k] = MetaEntry(v[0], v[1], v[2])
                    else:
                        # Shouldn't happen, but handle gracefully
                        result[k] = MetaEntry(v[0], v[1], time.time())
                return result
            return loaded_meta
        except Exception:
            return {}

    def _load_total_size(self, default: int = 0) -> int:
        if self.current_commit is None:
            return default
        total_bytes = self.long_term.get(TOTAL_VAR_SIZE_KEY % self.current_commit)
        if total_bytes is None:
            return default
        try:
            return pickle.loads(total_bytes)
        except Exception:
            return default

    def _find_all_commit_hashes(self) -> list[str]:
        """Find all commit hashes in the store by scanning for metadata keys."""
        meta_prefix = META_KEY.replace("%s", "")
        commit_hashes = []

        for key in self.long_term.keys():
            if key.startswith(meta_prefix):
                # Extract commit hash from key like "__meta__abc123"
                commit_hash = key.replace(meta_prefix, "")
                if commit_hash:
                    commit_hashes.append(commit_hash)

        return commit_hashes

    def _get_commit_created_at(self, commit_hash: str) -> float | None:
        """Get the creation timestamp for a commit, or None if unavailable."""
        meta = self._load_meta(commit_hash)
        if not meta:
            return None

        # All keys in a commit have the same created_at timestamp
        first_entry = next(iter(meta.values()), None)
        if first_entry:
            return first_entry.created_at

        return None

    def clean_orphans(self, min_age_seconds: int = 3600) -> int:
        """
        Remove orphaned commits unreachable from HEAD and older than min_age.

        Args:
            min_age_seconds: Only delete orphans older than this (default 1 hour)

        Returns:
            Number of orphaned commits cleaned
        """
        if self.current_commit is None:
            return 0

        # Mark phase: Find all reachable commits
        reachable = set(self.history())

        # Sweep phase: Find old orphaned commits
        cutoff_time = time.time() - min_age_seconds
        orphans_to_clean = []

        for commit_hash in self._find_all_commit_hashes():
            if commit_hash in reachable:
                continue

            created_at = self._get_commit_created_at(commit_hash)
            if created_at and created_at < cutoff_time:
                orphans_to_clean.append(commit_hash)

        # Delete orphaned commits
        for orphan_hash in orphans_to_clean:
            # Remove all keys associated with this orphaned commit
            keyset_key = COMMIT_KEYSET % orphan_hash
            keyset_bytes = self.long_term.get(keyset_key)
            if keyset_bytes:
                try:
                    keyset = pickle.loads(keyset_bytes)
                    # Remove all versioned data blobs
                    versioned_keys = list(keyset.values())
                    if versioned_keys:
                        self.long_term.remove_many(*versioned_keys)
                except Exception:
                    pass

            # Remove metadata for this commit
            self.long_term.remove_many(
                META_KEY % orphan_hash,
                COMMIT_KEYSET % orphan_hash,
                PARENT_COMMIT % orphan_hash,
                TOTAL_VAR_SIZE_KEY % orphan_hash,
            )

        return len(orphans_to_clean)
