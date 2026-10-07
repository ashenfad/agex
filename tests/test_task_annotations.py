"""Tasks under postponed annotations: a type defined in the function that
defines the task resolves, and one that can't be resolved is refused
when the task is made, never left unchecked."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from nontainer.conformance.corpus import calls

from agex import Agent
from agex.providers.scripted import ScriptedProvider


def test_types_defined_beside_the_task_resolve_and_are_bound():
    @dataclass
    class Score:
        student: str
        total: int

    @dataclass
    class Ranking:
        best: str
        scores: list[Score]

    agent = Agent(
        ScriptedProvider(
            [
                calls(
                    "run_python",
                    code=(
                        "scores = [Score(s, sum(v)) for s, v in answers.items()]\n"
                        "task.success(Ranking(best='ada', scores=scores))"
                    ),
                )
            ]
        )
    )

    @agent.task
    def rank(answers: dict[str, list[int]]) -> Ranking:
        """Rank the students."""

    assert set(rank.spec.types) == {"Score", "Ranking"}
    assert rank.spec.params["answers"].schema is not None
    ranking = rank({"ada": [3, 4]})
    assert ranking == Ranking(best="ada", scores=[Score("ada", 7)])
    with pytest.raises(TypeError, match="argument 'answers' must be"):
        rank({"ada": "34"})


def test_a_type_that_cannot_be_resolved_is_refused_when_the_task_is_made():
    agent = Agent(ScriptedProvider([]))
    with pytest.raises(TypeError, match="can't be resolved.*Later"):

        @agent.task
        def early() -> Later:
            """Too early."""

    @dataclass
    class Later:
        x: int
