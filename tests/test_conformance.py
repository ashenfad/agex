"""agex against nontainer's harness corpus: every scenario agex has the
capabilities for, with no known gaps."""

import pytest
from nontainer.conformance import applies, check, run
from nontainer.conformance.harness import SCENARIOS

from agex.conformance import AgexHarness

HARNESS = AgexHarness()


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda s: s.name)
def test_agex_honors_the_scenario(scenario):
    if not applies(scenario, HARNESS):
        missing = sorted(set(scenario.needs) - HARNESS.capabilities)
        pytest.skip(f"agex lacks {missing}")
    assert check(scenario, run(scenario, HARNESS)) == {}
