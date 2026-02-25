import numpy as np
from kvgit import Staged, Versioned

from agex.agent import Agent
from agex.eval.bridge import execute_sandboxed
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def test_class_serialization_with_method_calls():
    """
    Tests that a sandbox-defined class (StClass) and its instance (StInstance)
    survive a commit and can be used in subsequent phases.

    Verifies:
    - Class definitions persist as StClass across commits
    - Instances persist as StInstance across commits
    - Method calls work on rehydrated instances
    - numpy data inside instances survives serialization
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = Staged(Versioned(store), encoder=_agex_encoder, decoder=_agex_decoder)

    # Phase 1: Define a class, create an instance, call methods.
    program = """
import np

class DataProcessor:
    def __init__(self, name, data):
        self.name = name
        self.data = np.array(data)
        self.processed = False

    def process(self):
        self.data = self.data * 2
        self.processed = True
        return self.data

    def get_stats(self):
        if not self.processed:
            raise ValueError("Data not processed yet")
        return {
            'mean': float(np.mean(self.data)),
            'sum': float(np.sum(self.data)),
        }

processor = DataProcessor("dataset1", [1, 2, 3])
processor.process()
"""

    execute_sandboxed(program, agent, state)
    state.commit()

    # Phase 2: Call methods on the rehydrated instance.
    program2 = """
stats = processor.get_stats()
"""
    execute_sandboxed(program2, agent, state)

    stats = state.get("stats")

    # data was [1, 2, 3], process() makes it [2, 4, 6]
    # sum = 12, mean = 4.0
    assert stats["sum"] == 12
    assert stats["mean"] == 4.0


def test_function_and_class_across_phases():
    """
    Tests that top-level function definitions (StFunction) and class
    definitions (StClass) survive commits and can be used together
    in later phases.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = Staged(Versioned(store), encoder=_agex_encoder, decoder=_agex_decoder)

    # Phase 1: Define a helper function and a class.
    execute_sandboxed(
        """
import np

def scale(arr, factor):
    return arr * factor

class Accumulator:
    def __init__(self):
        self.total = 0
    def add(self, value):
        self.total = self.total + value
        return self.total
""",
        agent,
        state,
    )
    state.commit()

    # Phase 2: Use the function and class from phase 1.
    execute_sandboxed(
        """
arr = np.array([1, 2, 3])
scaled = scale(arr, 10)
acc = Accumulator()
acc.add(int(np.sum(scaled)))
""",
        agent,
        state,
    )
    state.commit()

    # Phase 3: Continue using both objects.
    execute_sandboxed(
        """
result = acc.add(100)
final_scaled = scale(scaled, 2)
final_sum = int(np.sum(final_scaled))
""",
        agent,
        state,
    )

    # scaled = [10, 20, 30], sum = 60; acc.add(60) -> 60, acc.add(100) -> 160
    assert state.get("result") == 160
    # final_scaled = [20, 40, 60], sum = 120
    assert state.get("final_sum") == 120
