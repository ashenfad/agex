"""agex against its shape corpus: every scenario, read back from its
committed JSON as another language would read it, on every rung agex
has; the JSON as the Python sources would write it; and a check that
catches what differs."""

import json
from dataclasses import dataclass, replace

import pytest
from nontainer.conformance.codec import dumps, load

from agex.conformance import AgexTasks
from agex.conformance.code_scenarios import CODE_SCENARIOS
from agex.conformance.export import JSON_DIR, files
from agex.conformance.runner import _same, applies, check, run
from agex.conformance.scenarios import (
    CAPABILITY,
    KEEPS,
    RANKING,
    RESUME_MISSING_LIVE,
    SCENARIOS,
    SCORE,
    TYPED_VALUE,
    ref,
)
from agex.conformance.shape import (
    LiveCall,
    PlaneExp,
    TaskScenario,
    call,
    restart,
    resume,
    success,
    task_success,
)
from agex.conformance.tasks import _json


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    return AgexTasks(tmp_path_factory.mktemp("shape"))


def committed(name: str) -> TaskScenario:
    return load(TaskScenario, json.loads((JSON_DIR / f"{name}.json").read_text()))


@pytest.mark.parametrize("rung", ["none", "process", "dud"])
@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agex_honors_the_shape_scenario(harness, scenario, rung):
    if rung not in harness.rungs:
        pytest.skip(f"agex has no {rung!r} rung here")
    if not applies(scenario, harness, rung):
        pytest.skip(f"doesn't apply on {rung!r}")
    assert check(scenario, run(committed(scenario.name), harness, rung)) == {}


def test_the_committed_corpus_is_what_the_sources_write():
    """Regenerate with ``python -m agex.conformance.export``."""
    for path, text in files().items():
        assert path.exists(), f"{path.name} is missing"
        assert path.read_text() == text, f"{path.name} is stale"
    assert {p.name for p in JSON_DIR.glob("*.json")} == {
        f"{s.name}.json" for s in [*SCENARIOS, *CODE_SCENARIOS]
    }


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_each_scenario_reads_back_from_its_json(scenario):
    assert load(TaskScenario, json.loads(dumps(scenario))) == scenario


# -- the check catches what differs -----------------------------------------------------


def expecting(scenario: TaskScenario, n: int, **changes) -> TaskScenario:
    expect = list(scenario.expect)
    expect[n] = replace(expect[n], **changes)
    return replace(scenario, expect=tuple(expect))


def test_the_check_names_each_part_that_differs(harness):
    seen = run(CAPABILITY, harness, "none")
    assert check(CAPABILITY, seen) == {}
    wrong = {
        "outcome1.status": {"status": "failed"},
        "outcome1.value": {"value": "BO"},
        "outcome1.asked": {"asked": 2},
        "outcome1.kept": {"kept": True},
        "outcome1.calls": {
            "calls": (LiveCall(input="directory", method="lookup", args=("bo",)),)
        },
    }
    for part, changes in wrong.items():
        assert set(check(expecting(CAPABILITY, 0, **changes), seen)) == {part}


def test_the_check_reads_the_plane_and_a_refusal(harness):
    kept = run(KEEPS, harness, "none")
    plane = PlaneExp(status="success", stored=("scores",), value=False)
    assert set(check(expecting(KEEPS, 0, plane=plane), kept)) == {"outcome1.plane"}
    refused = run(RESUME_MISSING_LIVE, harness, "none")
    assert check(RESUME_MISSING_LIVE, refused) == {}
    problems = check(
        expecting(RESUME_MISSING_LIVE, 1, names=("directory", "nope")), refused
    )
    assert (
        set(problems) == {"outcome2.names"} and "['nope']" in problems["outcome2.names"]
    )


def test_values_are_compared_as_json_of_their_declared_types():
    """A boolean is no number, and a record of another class isn't the
    record, however alike their JSON."""
    assert not _same(True, 1) and not _same({"n": 1}, {"n": True})
    assert _same(1, 1.0) and _same([1, [2]], (1, (2,)))

    @dataclass
    class Ranking:
        best: str
        scores: list

    defs = {"Ranking": RANKING, "Score": SCORE}
    assert _json(Ranking("ada", []), ref("Ranking"), defs) == {
        "best": "ada",
        "scores": [],
    }

    @dataclass
    class Other:
        best: str
        scores: list

    assert _json(Other("ada", []), ref("Ranking"), defs).startswith("!not Ranking")
    assert _json(True, {"type": "integer"}, defs).startswith("!not int")


def test_a_script_that_runs_out_is_caught(harness):
    short = replace(TYPED_VALUE, acts=(call(scores=[]),))
    problems = check(short, run(short, harness, "none"))
    assert "script" in problems and "outcome1.status" in problems


# -- the format refuses what can't be run -----------------------------------------------


def test_a_scenario_needs_one_expectation_per_call_and_resume():
    with pytest.raises(ValueError, match="1 call"):
        replace(TYPED_VALUE, expect=())


def test_a_scenario_starts_with_a_call():
    with pytest.raises(ValueError, match="first act must be a call"):
        replace(TYPED_VALUE, acts=(restart(), call(task_success(1))))


def test_a_resume_by_ref_after_a_restart_gives_an_outcome_to_resume():
    replace(
        TYPED_VALUE,
        acts=(call(), restart(), resume("yes", by="ref"), resume("and yes")),
        expect=(success(), success(), success()),
    )


def test_a_resume_by_outcome_after_a_restart_is_refused():
    with pytest.raises(ValueError, match="resume by ref"):
        replace(
            TYPED_VALUE,
            acts=(call(), restart(), resume("yes")),
            expect=(success(), success()),
        )
