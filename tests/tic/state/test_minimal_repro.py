import numpy as np

from tic.agent import Agent
from tic.eval.core import evaluate_program
from tic.state import Versioned
from tic.state.kv import Memory


def test_minimal_failure_repro():
    """
    A minimal test that reproduces the serialization failure.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory(as_bytes=True)
    state = Versioned(store)

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
    evaluate_program(phase_A, agent, state)
    state.snapshot()

    # Phase B: Reference the object from the previous phase in a new data structure.
    phase_B = """
d = {'proc': p1}
"""
    evaluate_program(phase_B, agent, state)

    # This snapshot will fail.
    state.snapshot()
