import numpy as np

from tic.agent import Agent
from tic.eval.core import evaluate_program
from tic.state import Versioned
from tic.state.kv import Memory


def test_class_serialization_with_closures():
    """
    Tests that a class instance captured in a lambda's closure is correctly
    deserialized and usable after a state snapshot.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = Versioned(store)

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
        # This lambda captures `self`. The serialization process needs to
        # correctly restore `self` as a `DataProcessor` instance.
        stat_lambda = lambda: {
            'mean': np.mean(self.data),
            'sum': np.sum(self.data),
        }
        return stat_lambda

processor = DataProcessor("dataset1", [1, 2, 3])
processor.process()
stats_func = processor.get_stats()
"""

    evaluate_program(program, agent, state)
    state.snapshot()

    # After the snapshot, retrieve the function.
    stats_func = state.get("stats_func")

    # This call should now succeed because the `self` in the closure is
    # correctly rehydrated into a TicInstance.
    stats = stats_func()

    # Verify the results
    # data was [1, 2, 3], process() makes it [2, 4, 6]
    # sum = 12, mean = 4.0
    assert stats["sum"] == 12
    assert stats["mean"] == 4.0
