"""Delegation: agex sessions as nontainer's delegates.

A session made with ``sessions=True`` can delegate through its
``sessions`` tool. Each delegate is a fork of the session's world, run
as a session of the same agent by :class:`Runner`, an async
:class:`nontainer.SessionRunner`: it opens the fork with the parent's
profile (a delegate holds no more than its parent), runs the task as the
delegate's first turn, and wakes it until nothing it waits on is
outstanding (:func:`nontainer.sessions.auntil_settled`). A delegate can
delegate in turn, the same way.

    chat = agent.session(ws, sessions=True)
    chat.say("ask a delegate to write the tests while you fix the bug")
    chat.close()   # waits for the delegates still running
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast

from nontainer import Answer, Profile, SessionRunner, Workspace
from nontainer.sessions import Sessions, auntil_settled

if TYPE_CHECKING:
    from .agent import Agent, Outcome

__all__ = ["MAX_WAKES", "Runner"]

MAX_WAKES = 10
"""How many woken turns a delegate gets, when ``ask`` sets no budget."""

T = TypeVar("T")


class _Ended(Exception):
    """A delegate's turn ended without a reply to settle on."""

    def __init__(self, answer: Answer) -> None:
        super().__init__(answer.text)
        self.answer = answer


def _failed(outcome: Outcome) -> Answer:
    said = f"{outcome.text}\n\n" if outcome.text else ""
    why = outcome.message or outcome.status
    return Answer(
        text=f"{said}[the delegate's turn ended {outcome.status}: {why}]",
        status="failed",
    )


async def _on_thread(call: Callable[[], T], undo: Callable[[T], Any]) -> T:
    """``call()`` on a worker thread. A thread can't be stopped, so a
    cancel waits for it to finish, then ``undo`` puts back what it made,
    and the cancel goes on: abandoned, the thread would leave a world
    open that nothing closes."""
    work = asyncio.ensure_future(asyncio.to_thread(call))
    cancelled = False
    while True:
        try:
            made = await asyncio.shield(work)
        except asyncio.CancelledError:
            if work.cancelled():
                raise
            cancelled = True
            continue
        except BaseException:
            if cancelled:
                raise asyncio.CancelledError from None
            raise
        break
    if cancelled:
        await asyncio.to_thread(undo, made)
        raise asyncio.CancelledError
    return made


def _as_runner(runner: Runner) -> SessionRunner:
    """``runner`` as the ``SessionRunner`` a ``Sessions`` takes: the
    protocol spells the synchronous ``run``, and an ``async def`` one is
    taken at runtime, which the protocol can't say."""
    return cast(SessionRunner, runner)


def _cancel_running(helper: Sessions) -> None:
    for job in helper.list():
        if job.status == "running":
            try:
                helper.cancel(job.name)
            except Exception:  # noqa: BLE001 - the rest still stop
                pass


class Runner:
    """Runs ``agent`` as the delegates of the session on ``ws``: an async
    :class:`nontainer.SessionRunner`, for ``Sessions(ws, Runner(agent,
    ws))``.

    Each delegate opens its fork with ``ws``'s profile, so it holds what
    its parent holds and no more, and is a session that can delegate
    too. Its answer is the reply it gives once nothing it waits on is
    outstanding: its own delegates have answered, and no note is
    queued. A turn that ends any other way than completed ends the
    delegate, ``failed``, saying how. ``budget``, when an ``ask`` passes
    one, is the most woken turns the delegate gets, ``max_wakes`` by
    default.

    A delegate that answers with its own delegates still out (its wakes
    spent) waits for them before its answer lands, since closing its
    helper joins their runs; a cancelled one cancels them.
    """

    def __init__(
        self, agent: Agent, ws: Workspace, *, max_wakes: int = MAX_WAKES
    ) -> None:
        store = ws.store
        if store is None:
            raise ValueError(
                f"session {ws.session!r} can't delegate: a delegate's world is a "
                "fork in its parent's store, and this workspace wasn't opened "
                "from a Store"
            )
        self.agent = agent
        self.store = store
        self.profile = Profile.of(ws)
        self.max_wakes = max_wakes

    def __repr__(self) -> str:
        return f"<Runner of {self.agent!r}>"

    async def run(self, session: str, task: str, *, budget: Any = None) -> Answer:
        from .agent import Session

        wakes = budget if isinstance(budget, int) else self.max_wakes
        child = await _on_thread(
            lambda: self.store.open(session, profile=self.profile),
            lambda ws: ws.close(),
        )
        try:
            helper = Sessions(
                child,
                _as_runner(Runner(self.agent, child, max_wakes=self.max_wakes)),
                loop=asyncio.get_running_loop(),
            )
            try:
                delegate = Session(self.agent, child, sessions=helper)

                async def run_turn(prompt: str | None) -> str:
                    stream = (
                        delegate.stream(prompt)
                        if prompt is not None
                        else delegate.stream(wake=True)
                    )
                    try:
                        outcome = await stream.wait()
                    except asyncio.CancelledError:
                        # the turn outlives its waiter: stop it, and let it
                        # land its run before the world closes under it
                        delegate.cancel()
                        await asyncio.wait({stream._start()})
                        raise
                    if outcome.status != "completed":
                        raise _Ended(_failed(outcome))
                    return outcome.text

                try:
                    settled = await auntil_settled(
                        run_turn,
                        helper,
                        delegate.inbox,
                        prompt=task,
                        max_wakes=wakes,
                    )
                except _Ended as ended:
                    return ended.answer
                except asyncio.CancelledError:
                    # stopped: its own delegates stop with it, rather than
                    # be waited for by a run nobody is waiting for
                    _cancel_running(helper)
                    raise
                return Answer(text=settled.text)
            finally:
                await helper.aclose()
        finally:
            await asyncio.to_thread(child.close)
