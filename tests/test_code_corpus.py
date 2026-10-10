"""agex against the shape corpus's code scenarios: tasks agent code
defines, read back from their committed JSON as another language would
read them, on every rung agex has."""

import json

import pytest
from nontainer.conformance.codec import load

from agex.conformance.code_scenarios import CODE_SCENARIOS
from agex.conformance.code_tasks import (
    AgexCodeTasks,
    CodeView,
    applies_code,
    check_code,
    run_code,
)
from agex.conformance.export import JSON_DIR
from agex.conformance.shape import CodeScenario, got


@pytest.fixture(scope="module")
def harness():
    return AgexCodeTasks()


def committed(name: str) -> CodeScenario:
    return load(CodeScenario, json.loads((JSON_DIR / f"{name}.json").read_text()))


@pytest.mark.parametrize("rung", ["none", "process", "dud"])
@pytest.mark.parametrize("scenario", CODE_SCENARIOS, ids=lambda s: s.name)
def test_agex_honors_the_code_scenario(harness, scenario, rung):
    if rung not in harness.rungs:
        pytest.skip(f"agex has no {rung!r} rung here")
    if not applies_code(scenario, harness, rung):
        pytest.skip(f"doesn't apply on {rung!r}")
    views = run_code(committed(scenario.name), harness, rung)
    assert check_code(scenario, views) == {}


@pytest.mark.parametrize("scenario", CODE_SCENARIOS, ids=lambda s: s.name)
def test_each_code_scenario_reads_back_from_its_json(scenario):
    assert committed(scenario.name) == scenario


def test_a_value_not_built_as_the_codes_own_class_is_caught():
    scenario = next(s for s in CODE_SCENARIOS if s.name.startswith("code-gets-its-own"))
    wrong = CodeView(status="value", value="!not the code's own Ranking: {...}")
    assert "act1.value" in check_code(scenario, [wrong])
    assert (
        check_code(
            scenario.__class__(**{**scenario.__dict__, "expect": (got(None),)}),
            [CodeView(status="value", value=None)],
        )
        == {}
    )
