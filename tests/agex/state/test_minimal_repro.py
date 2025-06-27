import numpy as np

from agex.agent import Agent
from agex.eval.core import evaluate_program
from agex.state import Versioned
from agex.state.kv import Memory


def test_minimal_failure_repro():
    """
    A minimal test that reproduces the serialization failure.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
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
