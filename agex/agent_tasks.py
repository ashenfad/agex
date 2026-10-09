"""Agent-defined tasks: functions agent code defines, run by a helper.

An embedder grants ``agex`` as a host object built per world:

    profile = Profile(python=PythonConfig(host_objects={
        "agex": agent_tasks(agent),
    }))

and agent code defines tasks with it, typed by their annotations:

    @agex.task
    def corners(shape: str) -> list[Point2D]:
        \"""The corner points of the named shape, counterclockwise.\"""

    pts = corners("unit square")

Agent code holds :class:`agex.stubs.AgexStub`. The stub compiles the
task's signature where the code runs and sends it as data
(:func:`nontainer.values.export_specs`), with each input encoded. The
host half, :class:`AgentTasks`, holds the embedder's agent and runs
nothing the code sent: it reads the spec back over classes built here
(:func:`nontainer.values.load_specs`), shapes with the same names and
fields, and runs a helper with them. The value comes back encoded, and
the stub decodes it into the caller's own classes.

**A helper is a delegate of the world that called.** It is asked
through the world's helper (:meth:`nontainer.sessions.Sessions.of`)
with an empty view, forked from the last commit, and a runner of its
own for the job (:class:`TaskRun`). So it is listed and kept with the
world's other delegates, and its answer is collected by the call, never
handed to the caller's model. A world with no helper open, or an
embedder that asks for it (``scratch=True``), puts helpers on scratch
worlds in memory instead.

A helper holds what its caller holds: the caller's profile, with the
inputs, ``task`` and the task's types added, and without ``agex``, so
helpers don't make helpers. Inputs and the value are values on every
rung (data, records, enums, bytes, tables and arrays); a live type is
refused where the task is defined.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import reprobate
from nontainer import (
    Answer,
    HostObject,
    JobRunning,
    Profile,
    SessionsError,
    Store,
    Workspace,
    values,
)
from nontainer.sessions import Sessions

from .agent import Agent, Outcome, _on_turn_loop
from .delegation import _on_thread
from .record import new_id
from .stubs import AgexStub, TaskStub
from .task import (
    PREVIEW,
    TaskHost,
    TaskSpec,
    ValueSpec,
    _begin,
    _brief,
    _schema,
    _TaskSession,
    _with_objects,
)

__all__ = ["MAX_PARALLEL", "AgentTasks", "TaskRun", "agent_tasks"]

MAX_PARALLEL = 4
"""How many helpers one ``.map`` runs at once, unless the embedder says."""

SHAPES = "shapes"
"""The module the helper's generated classes say they belong to."""

_NAME_TRIES = 8

_LANDING = 0.01
"""How often, in seconds, a call looks again for an answer still landing."""


def agent_tasks(
    agent: Agent, *, max_parallel: int = MAX_PARALLEL, scratch: bool = False
) -> HostObject:
    """``agex`` for a world's profile: agent code defines tasks with it,
    and ``agent`` runs them. ``max_parallel`` bounds how many helpers a
    ``.map`` runs at once; ``scratch=True`` puts every helper on a
    scratch world in memory rather than a delegate branch."""
    return HostObject(
        factory=lambda ws: AgentTasks(
            agent, ws, max_parallel=max_parallel, scratch=scratch
        ),
        stub=AgexStub,
    )


class _Called:
    """One call, read back: the task as a :class:`~agex.task.TaskSpec`
    over shapes, and its inputs decoded into them."""

    def __init__(
        self, data: Mapping[str, Any], inputs: Mapping[str, bytes], primer: str
    ) -> None:
        if not isinstance(data, Mapping) or data.get("format") != "agex-task/1":
            raise ValueError("not an agent-defined task")
        name, doc, params = data.get("name"), data.get("doc"), data.get("params")
        if not (
            isinstance(name, str)
            and name.isidentifier()
            and isinstance(doc, str)
            and isinstance(params, list)
            and all(isinstance(p, str) for p in params)
        ):
            raise ValueError("an agent-defined task that isn't well formed")
        try:
            specs = values.load_specs(data.get("specs"), module=SHAPES)
        except values.Malformed as error:
            raise ValueError(f"task {name!r}: {error}") from None
        if set(specs) != {*params, "return"} or set(inputs) != set(params):
            raise ValueError(f"task {name!r}: its inputs don't match its parameters")
        self.primer = primer
        self.inputs: dict[str, Any] = {}
        for key in params:
            try:
                self.inputs[key] = specs[key].decode(inputs[key])
            except (values.Mismatch, ValueError) as error:
                raise ValueError(f"{name}()'s argument {key!r}: {error}") from None
        found: dict[str, type] = {}
        for spec in specs.values():
            for tp in spec.types:
                found.setdefault(tp.__name__, tp)
        self.spec = TaskSpec(
            name=name,
            instructions=doc,
            params={key: _value_spec(specs[key]) for key in params},
            returns=_value_spec(specs["return"]),
            types=found,
        )

    def label(self) -> str:
        """The call as the rail shows it."""
        listed = ", ".join(
            f"{k}={reprobate.render(v, budget=80)}" for k, v in self.inputs.items()
        )
        return f"{self.spec.name}({listed})"


def _value_spec(spec: values.Spec) -> ValueSpec:
    return ValueSpec(
        annotation=spec.annotation,
        kinds=spec.kinds,
        spec=spec,
        schema=_schema(spec.annotation, spec.kinds),
    )


def _without_classes(profile: Profile, names: set[str]) -> Profile:
    """``profile`` without its classes named ``names``: in a helper's
    world the task's own classes take those names, since its values are
    built of them (the caller decodes them into its own classes)."""
    python = profile.python
    kept = tuple(c for c in python.classes if c.__name__ not in names)
    if len(kept) == len(python.classes):
        return profile
    return replace(profile, python=replace(python, classes=kept))


def _reply(status: str, message: str | None = None, value: bytes | None = None) -> dict:
    reply: dict[str, Any] = {"status": status}
    if message is not None:
        reply["message"] = message
    if value is not None:
        reply["value"] = value
    return reply


class TaskRun:
    """One agent-defined task's job: an async
    :class:`nontainer.SessionRunner` that runs a helper agent on the
    job's world, from the task's brief, until its code calls ``task``.

    It keeps the reply for the call that asked (:attr:`reply`): the
    value, encoded, or why there is none. nontainer's
    :class:`~nontainer.Answer` carries a summary, for the rail."""

    def __init__(
        self,
        agent: Agent,
        called: _Called,
        base: Profile,
        origin: str | None,
        open_world: Callable[[str, Profile], Workspace],
    ) -> None:
        self.agent = agent
        self.called = called
        self.base = base
        self.origin = origin
        self._open_world = open_world
        self.reply: dict[str, Any] = _reply("failed", "the helper never ran")
        self.done = threading.Event()
        """Set once the run is over, however it ended."""
        self._value: Any = None

    def __repr__(self) -> str:
        return f"<TaskRun {self.called.spec.name}>"

    def _helper(self) -> Agent:
        primer = "\n\n".join(p for p in (self.agent.primer, self.called.primer) if p)
        return Agent(
            self.agent.provider,
            primer=primer,
            settings=self.agent.settings,
            max_steps=self.agent.max_steps,
            compaction=self.agent.compaction,
        )

    async def run(
        self,
        session: str,
        task: str,
        *,
        budget: Any = None,
        forked_at: str | None = None,
    ) -> Answer:
        try:
            return await self._run(session)
        except Exception as error:
            self.reply = _reply("failed", f"{type(error).__name__}: {error}")
            raise
        finally:
            self.done.set()

    async def _run(self, session: str) -> Answer:
        spec = self.called.spec
        host = TaskHost(spec.returns.spec)
        entries = {
            key: HostObject(value, type=spec.params[key].spec)
            for key, value in self.called.inputs.items()
        }
        profile = _with_objects(
            _without_classes(self.base, set(spec.types)),
            {**entries, "task": HostObject(host, stub=TaskStub)},
            tuple(spec.types.values()),
        )

        def opened() -> Workspace:
            ws = self._open_world(session, profile)
            try:
                _begin(ws.provider.kv, spec, entries, self.origin)
            except BaseException:
                ws.close()
                raise
            return ws

        world = await _on_thread(opened, lambda ws: ws.close())
        try:
            helper = _TaskSession(self._helper(), world, host)
            stream = helper.stream(_brief(spec, self.called.inputs, {}))
            try:
                ran = await stream.wait()
            except asyncio.CancelledError:
                helper.cancel()
                await asyncio.wait({stream._start()})
                raise
        finally:
            await asyncio.to_thread(world.close)
        self.reply = self._settled(ran, host)
        return self._answer()

    def _settled(self, ran: Outcome, host: TaskHost) -> dict[str, Any]:
        result = host._result
        if result is None:
            why = ran.message or (
                "the run ended without a result"
                if ran.status == "completed"
                else ran.status
            )
            return _reply("failed", why)
        status, detail = result
        if status == "success":
            try:
                blob = values.encode(detail).to_bytes()
            except values.Unencodable as error:
                return _reply("failed", f"its value can't be sent back: {error}")
            self._value = detail
            return _reply("success", value=blob)
        return _reply(status, str(detail))

    def _answer(self) -> Answer:
        reply = self.reply
        name = self.called.spec.name
        if reply["status"] == "success":
            shown = reprobate.render(self._value, budget=PREVIEW)
            return Answer(text=f"{name} returned {shown}")
        if reply["status"] == "needs_input":
            return Answer(text=f"{name} asked: {reply.get('message')}", status="failed")
        return Answer(text=f"{name} failed: {reply.get('message')}", status="failed")


class AgentTasks:
    """The host half of ``agex`` for one world (``ws``): it runs the
    tasks that world's code defines, each call on a helper of its own
    (see :mod:`agex.agent_tasks`). Built by :func:`agent_tasks`'s
    factory as each world opens."""

    def __init__(
        self,
        agent: Agent,
        ws: Workspace,
        *,
        max_parallel: int = MAX_PARALLEL,
        scratch: bool = False,
    ) -> None:
        self._agent = agent
        self._ws = ws
        self._max_parallel = max_parallel
        self._scratch = scratch
        self._count = itertools.count(1)
        self._count_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"<agex for {self._ws.session!r}>"

    def held(self) -> list[str]:
        """The world's host objects, by name: what a task's parameters
        can't be named, since its helper holds them too."""
        return sorted(self._ws.runtime.python_config.host_objects)

    def call(
        self, spec: dict[str, Any], inputs: dict[str, bytes], primer: str
    ) -> dict[str, Any]:
        """Run the task ``spec`` once on ``inputs``; the reply."""
        return self.map(spec, [inputs], primer)[0]

    def map(
        self, spec: dict[str, Any], calls: list[dict[str, bytes]], primer: str
    ) -> list[dict[str, Any]]:
        """Run the task ``spec`` once per item of ``calls``, at once up to
        the limit; the replies, in order."""
        replies: list[dict[str, Any] | None] = [None] * len(calls)
        work: list[tuple[int, _Called]] = []
        for i, inputs in enumerate(calls):
            try:
                work.append((i, _Called(spec, inputs, primer)))
            except ValueError as error:
                replies[i] = _reply("failed", str(error))
        if work:
            base = self._base()
            helper = None if self._scratch else Sessions.of(self._ws)
            if helper is None:
                done = self._on_scratch(base, [c for _, c in work])
            else:
                done = self._as_delegates(helper, base, [c for _, c in work])
            for (i, _), reply in zip(work, done):
                replies[i] = reply
        return [r for r in replies if r is not None]

    # -- the helpers ----------------------------------------------------------------

    def _base(self) -> Profile:
        """The caller's profile without ``agex``: a helper doesn't make
        helpers."""
        profile = Profile.of(self._ws)
        python = profile.python
        kept = {
            name: entry
            for name, entry in python.host_objects.items()
            if not (isinstance(entry, HostObject) and entry.stub is AgexStub)
        }
        return replace(profile, python=replace(python, host_objects=kept))

    def _as_delegates(
        self, helper: Sessions, base: Profile, called: list[_Called]
    ) -> list[dict[str, Any]]:
        """Each call as a delegate of this world, ``max_parallel`` running
        at once; the replies, in order.

        Every fork is made here, on the calling thread: a host call runs
        on the thread running the caller's script, which holds its
        world's lock, and the fork takes that lock again. The runs go on
        on the helper's loop, and each answer is collected here, so the
        caller's model never sees it."""
        store = self._ws.store
        if store is None:
            raise RuntimeError("a world with a helper open was opened from a store")
        opener: Store = store
        runs = [
            TaskRun(
                self._agent,
                c,
                base,
                self._ws.session,
                lambda name, profile: opener.open(name, profile=profile),
            )
            for c in called
        ]
        started: list[tuple[str, TaskRun]] = []
        for run in runs:
            if len(started) >= self._max_parallel:
                started[len(started) - self._max_parallel][1].done.wait()
            started.append((self._ask(helper, run), run))
        for name, run in started:
            self._collect(helper, name, run)
        return [run.reply for run in runs]

    def _ask(self, helper: Sessions, run: TaskRun) -> str:
        """Ask ``helper`` for ``run``'s job, as a delegate branch named
        for the task; the job's name."""
        called = run.called
        for _ in range(_NAME_TRIES):
            with self._count_lock:
                n = next(self._count)
            try:
                job = helper.ask(
                    called.label(),
                    name=f"{called.spec.name}-{n}",
                    paths=[],
                    runner=run,
                )
                return job.name  # type: ignore[union-attr]
            except SessionsError as error:
                if "already exists" not in str(error):
                    raise
        job = helper.ask(
            called.label(),
            name=f"{called.spec.name}-{new_id()[:8]}",
            paths=[],
            runner=run,
        )
        return job.name  # type: ignore[union-attr]

    @staticmethod
    def _collect(helper: Sessions, name: str, run: TaskRun) -> None:
        """Wait for job ``name`` to land, and collect its answer: what
        the task said is in ``run.reply``, and the answer, read, is never
        handed to the caller's model as a note."""
        run.done.wait()
        while True:
            try:
                helper.result(name)
                return
            except JobRunning:
                time.sleep(_LANDING)  # the run is over; its answer is landing
            except SessionsError as error:
                run.reply = _reply("failed", str(error))
                return

    def _on_scratch(self, base: Profile, called: list[_Called]) -> list[dict[str, Any]]:
        """Each call on a scratch world of its own, in memory,
        ``max_parallel`` at once; the replies, in order."""

        def one(c: _Called) -> dict[str, Any]:
            store = Store(memory=True)
            try:
                run = TaskRun(
                    self._agent,
                    c,
                    base,
                    None,
                    lambda name, profile: store.open(name, profile=profile),
                )
                name = f"{c.spec.name}-{new_id()[:8]}"
                _on_turn_loop(run.run(name, c.label()), "agex task")
                return run.reply
            finally:
                store.close()

        if len(called) == 1:
            return [one(called[0])]
        # the pool's threads run in the caller's context, which says
        # which loop the turn is on
        context = contextvars.copy_context()
        with ThreadPoolExecutor(
            max_workers=min(self._max_parallel, len(called)),
            thread_name_prefix="agex-scratch",
        ) as pool:
            return list(pool.map(lambda c: context.copy().run(one, c), called))
