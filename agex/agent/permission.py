"""Public, host-facing types for the permission interrupt/resume flow.

A task suspends by calling ``task_request_permission(scope, reason)`` in its
code; at the boundary the loop raises :class:`PermissionPending` to the host.
The host decides and resumes via ``task.resume(session=..., response=...)``
with a :class:`PermissionResponse`.

``PermissionResponse`` is deliberately minimal for v1 (``granted`` + an
optional free-text ``note``). It is *not* a bare bool so that fields like
``constraint``/``expires`` can be added later, with their enforcement, without
a breaking change. ``note`` is not a control field — it needs no enforcement;
it's most useful on a denial, to guide the agent's pivot on resume.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PermissionResponse:
    """The host's decision on a permission request.

    Session-free: the session is supplied alongside it to ``task.resume`` (the
    caller always has it from the access path), consistent with "events/objects
    don't carry session".
    """

    granted: bool
    note: str | None = None


class PermissionPending(Exception):
    """Raised to the host when a task suspends awaiting a capability grant.

    Catch it, decide, and resume::

        try:
            df = clean_data(df, session="abc")
        except PermissionPending as p:
            df = clean_data.resume(session="abc", response=p.respond(granted=True))

    ``scopes`` is the set of requested capability scopes (one or more). An
    atomic ``respond(granted=...)`` grants or denies the whole set.
    """

    def __init__(
        self, *, scopes: set[str], task_name: str, reason: str | None = None
    ) -> None:
        self.scopes = scopes
        self.task_name = task_name
        self.reason = reason
        msg = f"task {task_name!r} is awaiting grants for {sorted(scopes)!r}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)

    def respond(self, granted: bool, note: str | None = None) -> "PermissionResponse":
        """Mint a :class:`PermissionResponse` for this request."""
        return PermissionResponse(granted=granted, note=note)
