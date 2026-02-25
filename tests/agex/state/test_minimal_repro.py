import numpy as np
from kvgit import Staged, Versioned

from agex.agent import Agent
from agex.eval.bridge import execute_sandboxed
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def test_minimal_failure_repro():
    """
    Verifies that a sandbox-defined class instance (StInstance) can be
    created in one phase, referenced inside a dict in a second phase,
    and that committing the state succeeds in both cases.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = Staged(Versioned(store), encoder=_agex_encoder, decoder=_agex_decoder)

    # Phase A: Define a class and create an instance.
    phase_A = """
import np
class MyProc:
    def __init__(self, data):
        self.data = data
    def process(self):
        return np.sum(self.data)

p1 = MyProc([1,2,3])
"""
    execute_sandboxed(phase_A, agent, state)
    state.commit()

    # Phase B: Reference the object from the previous phase in a new data structure.
    phase_B = """
d = {'proc': p1}
"""
    execute_sandboxed(phase_B, agent, state)

    # This commit should succeed -- the StInstance inside the dict is pickleable.
    state.commit()
