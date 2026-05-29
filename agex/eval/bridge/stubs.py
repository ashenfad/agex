"""Stand-ins for scoped-but-ungranted registrations.

When a registration is gated by a ``scope=`` that the current session has
not been granted, agex registers one of these stand-ins into the sandtrap
``Policy`` *instead of* the real fn / class / module. sandtrap permits the
stand-in (it is an ordinary registered member), but using it raises
:class:`~agex.eval.error.ScopeRequired` — which the agent sees as a normal
error observation and reacts to by calling ``task_request_permission``.

The error is built here, at policy-build time, where the scope is known —
so the agent gets an instructional message without any error-interception.

sandtrap-gate notes (see the spike test ``test_scope_stubs.py``):

* fn stub  → a plain function; raising on call propagates as an ordinary
  exception, caught and fed back to the agent.
* cls stub → registered ``constructable=True`` so its raising ``__init__``
  (not a generic gate denial) is what surfaces.
* module stub → registered *by identity* with permissive ``include`` so the
  attribute gate proceeds to ``getattr`` → ``__getattr__`` (which raises).
  Covers both ``mod.member`` and ``from mod import member``. Dunder access
  falls through to ``AttributeError`` so sandtrap's internal probing of the
  object doesn't trip ``ScopeRequired`` prematurely.
"""

from __future__ import annotations

from typing import Any

from agex.eval.error import ScopeRequired


def _message(name: str, scope: str) -> str:
    return (
        f"`{name}` requires the `{scope}` scope, which is not granted in this "
        f"session. Call task_request_permission(scope='{scope}') to request it."
    )


def make_fn_stub(name: str, scope: str):
    """A function stand-in that raises :class:`ScopeRequired` when called."""

    def _stub(*args: Any, **kwargs: Any):
        raise ScopeRequired(_message(name, scope), scope=scope, name=name)

    _stub.__name__ = name
    _stub.__qualname__ = name
    return _stub


def make_cls_stub(name: str, scope: str) -> type:
    """A class stand-in that raises :class:`ScopeRequired` on construction."""

    def _init(self: Any, *args: Any, **kwargs: Any):
        raise ScopeRequired(_message(name, scope), scope=scope, name=name)

    return type(name, (), {"__init__": _init})


def make_module_stub(name: str, scope: str):
    """A module/object stand-in whose attribute access raises ``ScopeRequired``.

    Register by identity with permissive ``include`` (the *scope* is the gate,
    not include/exclude).
    """
    mod_name = name

    class _ModuleStub:
        def __getattr__(self, attr: str):
            # Let dunder/internal probing fall through so sandtrap's own
            # handling of the object doesn't trip ScopeRequired.
            if attr.startswith("__") and attr.endswith("__"):
                raise AttributeError(attr)
            member = f"{mod_name}.{attr}"
            raise ScopeRequired(_message(member, scope), scope=scope, name=member)

    return _ModuleStub()
