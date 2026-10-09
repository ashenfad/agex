"""agex under nontainer's harness corpus.

:class:`AgexHarness` runs a session with a
:class:`~agex.providers.scripted.ScriptedProvider` whose script is the
conformance runner's clock, so the corpus checks agex's loop against the
same contract the agno adapter is checked against:

    from nontainer.conformance import check, run
    from nontainer.conformance.harness import SCENARIOS

    harness = AgexHarness()
    for scenario in SCENARIOS:
        print(scenario.name, check(scenario, run(scenario, harness)))
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from nontainer import Store, Workspace
from nontainer.adapters.corpus_delegates import CorpusDelegates
from nontainer.compaction import Policy
from nontainer.conformance.corpus import Scenario
from nontainer.conformance.runner import Clock, RunView
from nontainer.turns import TurnEvent

from ..agent import CLOSING_NOTE, Agent, Session
from ..providers.scripted import ScriptedProvider
from ..record import Run, ToolResult

__all__ = ["AgexHarness", "AgexSession"]


def _closes_early(run: Run) -> bool:
    last = run.messages[-1] if run.messages else None
    return (
        last is not None
        and last.role == "assistant"
        and (last.text.startswith(CLOSING_NOTE))
    )


class AgexSession:
    """One workspace, driven by an agex session over the clock's script."""

    def __init__(
        self,
        ws: Workspace,
        clock: Clock,
        budget: int | None = None,
        sessions: Any = None,
    ) -> None:
        policy = None if budget is None else Policy(budget=budget)
        agent = Agent(ScriptedProvider(clock.next), compaction=policy)
        self.session: Session = agent.session(ws, sessions=sessions)
        self.inbox = self.session.inbox

    def turn(self, prompt: str) -> list[TurnEvent]:
        async def collect() -> list[TurnEvent]:
            return [event async for event in self.session.stream(prompt)]

        return asyncio.run(collect())

    def resume(self) -> list[TurnEvent]:
        async def collect() -> list[TurnEvent]:
            return [event async for event in self.session.stream(resume=True)]

        return asyncio.run(collect())

    def wake(self) -> list[TurnEvent]:
        async def collect() -> list[TurnEvent]:
            return [event async for event in self.session.stream(wake=True)]

        return asyncio.run(collect())

    def cancel(self) -> None:
        self.session.cancel()

    def runs(self) -> list[RunView]:
        return [
            RunView(
                closing_note=_closes_early(run),
                tool_results=sum(
                    isinstance(part, ToolResult)
                    for message in run.messages
                    for part in message.parts
                ),
            )
            for run in self.session.runs
        ]

    def run_statuses(self, info: Mapping[str, Any]) -> tuple[str, ...]:
        runs = info.get("runs")
        if not isinstance(runs, Mapping):
            return ()
        return tuple(str(status) for status in runs.values())

    def close(self) -> None:
        pass


class AgexHarness:
    """agex as a conformance harness."""

    name = "agex"

    def __init__(self) -> None:
        self.capabilities: frozenset[str] = frozenset(
            {"resume", "keeps-aborted-runs", "compaction", "delegation"}
        )
        self.known_gaps: dict[str, dict[str, str]] = {}

    def open(
        self,
        ws: Workspace,
        clock: Clock,
        *,
        budget: int | None = None,
        sessions: Any = None,
    ) -> AgexSession:
        return AgexSession(ws, clock, budget, sessions)

    def delegation(self, store: Store, scenario: Scenario) -> CorpusDelegates:
        """What runs a scenario's delegates: agex sessions too."""
        return CorpusDelegates(self, store, scenario)
