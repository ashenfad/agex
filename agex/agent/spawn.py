"""In-agent ephemeral clones — the ``spawn`` object injected into agent code.

See ``roadmap/spawn.md``. ``spawn`` lets agent code define and run ephemeral,
memoryless clones of the parent agent to fulfill typed subtasks, with a
blocking, ``concurrent.futures``-shaped surface (so the agent reaches for
``submit`` / ``.result()`` / ``map`` — the API it already knows — and never
touches ``async``/``await``):

    @spawn.task
    def gen_svg(prompt: str) -> Resource: ...

    svg   = gen_svg("a castle")                 # direct call: blocks, returns Resource
    h     = spawn.submit(gen_svg, "a castle")   # -> Future[Resource]
    tiles = spawn.map(gen_svg, ["a", "b"])      # -> list[Resource]

Clones run on fresh ephemeral ``Live`` state (no kvgit, blank VFS/cache),
in-process, with ``spawn`` stripped from their own namespace (depth-1 leaf
workers). Concurrency is a per-emission thread pool bounded by the parent's
``max_spawns``; it never uses the parent's event loop.

Implementation notes:
- Every *agent-facing* method/attribute here is public (no leading underscore)
  so sandtrap permits the access (an unregistered object allows public attrs;
  see ``sandtrap/policy.py``). Internals are underscore-prefixed and therefore
  unreachable from sandboxed code.
- Clone events forward to the parent's ``on_event`` by default, labeled via a
  ``Namespaced`` state tag for stream demux. Tokens are opt-in. We deliberately
  do NOT route clone events into the parent's durable log (stream, don't store).
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import itertools
from typing import Any, Callable, Iterable

# Typing-facing alias for the handle returned by ``spawn.submit``. At runtime it
# is an ordinary ``concurrent.futures.Future``; the public Future API
# (``result``/``done``/``exception``/...) is all sandbox-callable.
SpawnHandle = concurrent.futures.Future


def _nested_sandbox_classes(annotation: Any, st_class: type) -> list:
    """Find sandbox-defined classes (``StClass``) nested inside a generic
    annotation (``list[Tile]``, ``dict[str, Tile]``, ...), recursively."""
    from typing import get_args

    found = []
    for arg in get_args(annotation):
        if isinstance(arg, st_class):
            found.append(arg)
        found.extend(_nested_sandbox_classes(arg, st_class))
    return found


def _collect_seed_classes(task_wrapper: Any) -> dict[str, bytes]:
    """Collect sandbox-defined classes referenced in a spawn task's signature
    as inactive pickle bytes, keyed by class name.

    These are ``StClass`` wrappers (sandtrap) for classes the agent defined in
    its own sandbox. A spawn clone is a *separate* sandbox without them, so they
    must be reconstructed inside the clone. Pickling an ``StClass`` strips its
    active binding (``__getstate__`` clears ``_compiled_cls``/``_gates``), so the
    bytes reconstruct an *inactive* wrapper the clone re-binds to its own gates.

    v1 scope: the **top-level** return type. A sandbox class nested in a generic
    return type (``list[Tile]``) can't be reconstructed in the clone yet, so we
    **fail fast** with a clear message at definition time rather than let the
    clone ``NameError`` into a 10-iteration timeout. Parameter types already
    cross as instances (read-only, duck-typed) and need no seeding. Full
    generic/param support is a follow-up (see roadmap/spawn-type-sharing.md and
    roadmap/unwrapped.md).
    """
    import pickle

    try:
        from sandtrap.wrappers import StClass
    except Exception:
        return {}

    seed: dict[str, bytes] = {}
    rt = getattr(task_wrapper, "_return_type", None)
    if isinstance(rt, StClass):
        name = getattr(rt, "_name", None) or getattr(rt, "__name__", None)
        if name:
            try:
                seed[name] = pickle.dumps(rt)
            except Exception:
                pass  # unpicklable class → skip; the clone NameErrors as before
        return seed

    # Sandbox class nested in a generic return type → not supported yet; fail
    # fast (a runtime NameError in the clone would otherwise burn the iteration
    # budget and surface as an opaque TaskTimeout).
    nested = _nested_sandbox_classes(rt, StClass)
    if nested:
        names = ", ".join(sorted(getattr(c, "_name", None) or "?" for c in nested))
        task_name = getattr(task_wrapper, "_task_name", None) or "?"
        raise TypeError(
            f"spawn task '{task_name}': return type references sandbox-defined "
            f"class(es) [{names}] nested in a generic, which spawn can't yet "
            f"reconstruct in the clone. Return the class directly (e.g. "
            f"'-> Tile'), or use a registered/builtin type."
        )
    return seed


class SpawnTaskWrapper:
    """What ``@spawn.task`` returns: a blocking callable that runs an ephemeral
    clone loop and returns the typed result. Also accepted by
    ``spawn.submit`` / ``spawn.map`` (identified by ``_is_spawn_task``)."""

    _is_spawn_task = True

    def __init__(self, spawn: "Spawn", task_wrapper: Any):
        self._spawn = spawn
        self._task = task_wrapper  # the clone's TaskWrapper
        # Introspection passthrough so the agent can read the signature/doc.
        self.__name__ = getattr(task_wrapper, "__name__", "spawn_task")
        self.__doc__ = getattr(task_wrapper, "__doc__", None)
        sig = getattr(task_wrapper, "__signature__", None)
        if sig is not None:
            self.__signature__ = sig
        # Sandbox-defined classes referenced in the signature must be made
        # available *inside* the clone (a separate sandbox). Capture them as
        # inactive pickle bytes now; each invocation reconstructs a fresh
        # inactive copy and seeds it into the clone's namespace, where the
        # clone's auto-activation binds it to the clone's gates. v1 scope:
        # the top-level return type. See roadmap/spawn-type-sharing.md.
        self._seed_classes: dict[str, bytes] = _collect_seed_classes(task_wrapper)

    def __call__(self, *args: Any) -> Any:
        # Direct call: run the clone loop inline (blocking) on the calling thread.
        return self._spawn._run_one(self, args)

    def __repr__(self) -> str:
        return f"<spawn task {self.__name__!r}>"


class Spawn:
    """The ``spawn`` object injected into an agent's ``__main__`` namespace.

    One instance per emission. Holds a reference to the parent's cached clone
    template and the current emission's ``on_event``; reads the live session /
    handlers from contextvars at call time.
    """

    def __init__(
        self,
        clone: Any,
        parent_agent: Any,
        on_event: Callable[[Any], None] | None = None,
        *,
        stream_tokens: bool = False,
    ):
        self._clone = clone
        self._parent = parent_agent
        self._on_event = on_event
        self._stream_tokens = stream_tokens
        self._pool: concurrent.futures.ThreadPoolExecutor | None = None
        self._counter = itertools.count()

    # ------------------------------------------------------------------ #
    # Public surface (agent-facing)                                      #
    # ------------------------------------------------------------------ #

    def task(self, fn: Callable | None = None, /, **_kwargs: Any) -> Any:
        """Define a spawn task. Decorator factory — usable bare (``@spawn.task``)
        or called (``@spawn.task(...)``). Keyword arguments are reserved for
        future use (e.g. ``fs=`` for read-only VFS mounts) and ignored in v1.

        The decorated function's signature, docstring, and return annotation are
        the contract the clone must satisfy — identical to ``@agent.task``.
        """

        def decorate(func: Callable) -> SpawnTaskWrapper:
            task_wrapper = self._clone.task(func)
            return SpawnTaskWrapper(self, task_wrapper)

        if fn is None:
            return decorate
        return decorate(fn)

    def submit(self, spawn_task: Any, *args: Any) -> "concurrent.futures.Future":
        """Run a ``@spawn.task`` on a worker thread; return a Future handle.
        Collect with ``.result()`` (or via ``spawn.map``). Concurrent submits
        run in parallel up to the agent's ``max_spawns``."""
        task = self._resolve_task(spawn_task, "submit")
        idx = next(self._counter)
        ctx = contextvars.copy_context()  # carry session/handlers into the thread

        def work() -> Any:
            return ctx.run(self._run_one, task, args, idx)

        return self._ensure_pool().submit(work)

    def map(self, spawn_task: Any, iterable: Iterable) -> list:
        """Run a ``@spawn.task`` once per item, concurrently; return results in
        order. Mirrors ``executor.map`` — raises on the first failing item."""
        # submit() resolves/validates the task; iterate first so a bad task
        # raises immediately rather than mid-iteration.
        items = list(iterable)
        handles = [self.submit(spawn_task, x) for x in items]
        return [h.result() for h in handles]

    # ------------------------------------------------------------------ #
    # Internals (underscore — not reachable from sandboxed code)         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_task(spawn_task: Any, where: str) -> "SpawnTaskWrapper":
        """Return the underlying ``SpawnTaskWrapper`` for a task passed back
        from agent code. Sandtrap wraps a host callable returned into the
        sandbox in an ``StFunction`` (the original sits on ``._compiled``), so
        when the agent passes the task to ``spawn.submit``/``map`` we receive
        the wrapper, not the original — unwrap it here."""
        if getattr(spawn_task, "_is_spawn_task", False):
            return spawn_task
        inner = getattr(spawn_task, "_compiled", None)
        if getattr(inner, "_is_spawn_task", False):
            return inner
        raise TypeError(
            f"spawn.{where}() expects a @spawn.task function as its first "
            f"argument, got {type(spawn_task).__name__}."
        )

    def _ensure_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._pool is None:
            n = max(1, int(getattr(self._parent, "max_spawns", 8)))
            self._pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=n, thread_name_prefix="agex-spawn"
            )
        return self._pool

    def _run_one(self, spawn_task: SpawnTaskWrapper, args: tuple, idx: int = 0) -> Any:
        """Run one clone task loop synchronously on fresh, labeled ephemeral
        state. Forwards the parent session/handlers (from contextvars) and
        converts a sub-task failure into a recoverable ``EvalError`` (mirroring
        ``_wrap_sub_agent_task``), so the parent sees an error observation rather
        than a terminal signal."""
        from agex.agent.datatypes import TaskClarify, TaskFail, _TaskPending
        from agex.eval.bridge.policy import (
            _current_on_event,
            _current_on_token,
            _current_session,
        )
        from agex.eval.error import EvalError
        from agex.state import Namespaced
        from agex.state.live import Live

        task = spawn_task._task
        name = task._task_name

        # Validate the inputs against the task signature (reuse the wrapper's
        # binder); ignore the session/handlers it defaults — we use contextvars.
        inputs_instance, _, _, _ = task._bind_and_validate(*args)

        session = _current_session.get()
        on_event = self._on_event
        if on_event is None:
            on_event = _current_on_event.get()
        on_token = _current_on_token.get() if self._stream_tokens else None

        # Per-invocation labeled, non-versioned state for event-stream demux.
        state = Namespaced(Live(), f"spawn:{name}:{idx}")

        # Seed sandbox-defined signature classes into the clone (fresh inactive
        # copy per invocation; the clone's auto-activation binds them). Pass the
        # seeded copy as the return type so validation matches the clone's own
        # reconstruction rather than the parent's distinct class object.
        import pickle

        return_type = task._return_type
        if spawn_task._seed_classes:
            seed = {nm: pickle.loads(b) for nm, b in spawn_task._seed_classes.items()}
            state["__setup_namespace__"] = seed
            rt_name = getattr(return_type, "_name", None) or getattr(
                return_type, "__name__", None
            )
            if rt_name in seed:
                # Validate against the freshly-seeded class the clone actually
                # constructs from: the returned StInstance carries this exact
                # StClass as its _st_class, so validation is a real identity
                # check (see validate_with_sampling's StClass branch), not the
                # Pydantic "arbitrary type, allow anything" no-op.
                return_type = seed[rt_name]

        try:
            return self._clone._run_task_loop(
                task_name=name,
                docstring=task._effective_docstring,
                inputs_dataclass=task._inputs_dataclass,
                inputs_instance=inputs_instance,
                return_type=return_type,
                state=state,
                session=session,
                on_event=on_event,
                on_token=on_token,
                setup=task._setup,
                on_conflict="retry",
                max_conflict_retries=task._max_conflict_retries,
                emit_task_start=True,
            )
        except TaskFail as e:
            raise EvalError(f"Spawned task '{name}' failed: {e.message}") from None
        except TaskClarify as e:
            raise EvalError(
                f"Spawned task '{name}' needs clarification: {e.message}"
            ) from None
        except _TaskPending as e:
            # Ephemeral clones cannot durably suspend for a human grant.
            scopes = ", ".join(sorted(e.scopes))
            raise EvalError(
                f"Spawned task '{name}' requires capability scope(s) [{scopes}] "
                f"which cannot be granted to an ephemeral clone."
            ) from None

    def close(self) -> None:
        """Shut the thread pool down (bounds threads to one emission)."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
