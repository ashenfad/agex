"""Session-scoped capability grants.

The set of scopes granted in a session lives under a host-private *versioned*
key (``__grant_set__``): it commits with the rest of session state (durable,
rolls back with undo, survives a restart on versioned state), and ``__``-prefix
filtering keeps it invisible to and unwritable by sandboxed agent code.

``scopes(state)`` is the accessor, mirroring agex's existing ``view(state)`` /
``events(state)`` free-function idiom — the session is whatever
``agent.state(session)`` you pass in.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

GRANT_SET_KEY = "__grant_set__"


def read_grants(state: MutableMapping[str, Any]) -> set[str]:
    """Read the granted-scope set from a session's state (empty when absent).

    Used by the policy build to compute the per-execution effective policy.
    """
    current = state.get(GRANT_SET_KEY)
    return set(current) if current else set()


class ScopeSet:
    """Grant / revoke / query the capability scopes granted in a session.

    Obtained via :func:`scopes`. Grants are a deliberate authorization act, so
    a standalone ``grant``/``revoke`` commits immediately on versioned state
    (mirrors a host-side file write) — the next task then sees it. On Live
    (ephemeral) state there is no commit; the write is already in memory.
    """

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self._state = state

    def list(self) -> set[str]:
        """The scopes currently granted in this session."""
        return read_grants(self._state)

    def has(self, scope: str) -> bool:
        return scope in self.list()

    def grant(self, scope: str) -> None:
        current = self.list()
        if scope in current:
            return
        current.add(scope)
        self._apply(current, granted=[scope])

    def revoke(self, scope: str) -> None:
        current = self.list()
        if scope not in current:
            return
        current.discard(scope)
        self._apply(current, revoked=[scope])

    def _apply(
        self,
        scope_set: set[str],
        *,
        granted: list[str] | None = None,
        revoked: list[str] | None = None,
    ) -> None:
        from agex.agent.loop.event_factories import create_permission_event
        from agex.state import commit_state, is_live_root
        from agex.state.log import add_event_to_log

        self._state[GRANT_SET_KEY] = scope_set
        # A standalone grant/revoke is a host action — emit the PermissionEvent
        # notification (agent_name="System", like SystemNoteEvent) so it's
        # narrated/audited, mirroring how a file write emits a FileEvent. (The
        # request/resume flow emits its own event with the real agent name.)
        add_event_to_log(
            self._state,
            create_permission_event(
                "System", granted=granted or [], revoked=revoked or []
            ),
        )
        if not is_live_root(self._state):
            commit_state(self._state)


def scopes(state: MutableMapping[str, Any]) -> ScopeSet:
    """Return the grant accessor for a session's state.

    Mirrors ``view(state)`` / ``events(state)``::

        from agex import scopes
        scopes(agent.state("alice")).grant("email")
    """
    return ScopeSet(state)
