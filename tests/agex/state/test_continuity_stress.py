import numpy as np
from gitkv import Staged, Versioned

from agex.agent import Agent
from agex.eval.bridge import execute_sandboxed
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory


def _make_versioned(store=None, commit_hash=None):
    """Create a Staged store wrapping a gitkv Versioned with agex codecs."""
    if store is None:
        store = Memory()
    kw = {}
    if commit_hash is not None:
        kw["commit_hash"] = commit_hash
    return Staged(
        Versioned(store, **kw),
        encoder=_agex_encoder,
        decoder=_agex_decoder,
    )


def test_multi_step_program_with_snapshots():
    """
    Tests that a series of programs, separated by snapshots, can build upon
    each other's state, correctly serializing and rehydrating StFunctions,
    StClasses, StInstances, and numpy arrays.
    """
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = _make_versioned(store)

    # Turn 1: Define a function and import numpy
    execute_sandboxed(
        "import np\ndef my_func(x):\n    return x * 2\na = 10", agent, state
    )
    state.commit()

    # Turn 2: Use the function and create a numpy array
    execute_sandboxed("b = my_func(a)\narr = np.array([1, b])", agent, state)
    state.commit()

    # Turn 3: Define a class that uses the array
    execute_sandboxed(
        """
class MyClass:
    def __init__(self, val):
        self.val = val
    def get_val(self):
        return self.val * arr[0]

instance = MyClass(arr[1])
""",
        agent,
        state,
    )
    state.commit()

    # Turn 4: Use the class instance
    execute_sandboxed("c = instance.get_val()", agent, state)
    state.commit()

    # Turn 5: Define and call another function using cross-phase state
    execute_sandboxed("def add_to(x):\n    return x + c\nd = add_to(5)", agent, state)
    state.commit()

    # Verify the final state
    assert state.get("d") == 25
    final_arr = state.get("arr")
    np.testing.assert_array_equal(final_arr, np.array([1, 20]))

    # Load the state into a new agent and verify it still works
    state2 = _make_versioned(store, commit_hash=state.current_commit)
    agent2 = Agent()
    agent2.module(np, name="np")

    # Run a new program that uses the restored state
    execute_sandboxed("e = my_func(d)\nf = add_to(a)", agent2, state2)
    assert state2.get("e") == 50
    assert state2.get("f") == 30


def test_comprehensive_serialization_stress():
    """
    Multi-phase stress test for the sandtrap serialization system.

    Tests state continuity across 5 phases covering:
    - Top-level function definitions (StFunction) surviving commits
    - Class definitions (StClass) and instances (StInstance) surviving commits
    - numpy arrays persisting across phases
    - Cross-phase references (functions using data from earlier phases)
    - Large data structures
    - Multiple rehydration cycles with different agents
    """
    agent = Agent(eval_timeout_seconds=15.0)
    agent.module(np, name="np")

    store = Memory()
    state = _make_versioned(store)

    # Phase 1: Function definitions and basic data
    phase1 = """
import np

def multiply(x, factor):
    return x * factor

def create_array(n):
    return np.arange(n)

base_value = 5
result1 = multiply(base_value, 10)
arr1 = create_array(100)
"""

    execute_sandboxed(phase1, agent, state)
    state.commit()

    assert state.get("result1") == 50
    assert len(state.get("arr1")) == 100

    # Phase 2: Classes with methods and numpy integration
    phase2 = """
class DataProcessor:
    def __init__(self, name, data):
        self.name = name
        self.data = np.array(data)
        self.processed = False

    def process(self):
        self.data = multiply(self.data, 2)
        self.processed = True
        return self.data

    def stats(self):
        return {
            'mean': float(np.mean(self.data)),
            'sum': float(np.sum(self.data)),
            'max': float(np.max(self.data)),
            'name': self.name,
        }

processor1 = DataProcessor("dataset1", [1, 2, 3, 4, 5])
data1 = processor1.process()
processor2 = DataProcessor("dataset2", [10, 20, 30])
data2 = processor2.process()
"""

    execute_sandboxed(phase2, agent, state)
    state.commit()

    # Verify phase 2 by calling methods on committed instances
    execute_sandboxed(
        """
stats1 = processor1.stats()
stats2 = processor2.stats()
""",
        agent,
        state,
    )
    state.commit()

    stats1 = state.get("stats1")
    stats2 = state.get("stats2")
    assert stats1["sum"] == 30  # [2, 4, 6, 8, 10] sum = 30
    assert stats1["name"] == "dataset1"
    assert stats2["sum"] == 120  # [20, 40, 60] sum = 120

    # Phase 3: Nested classes and cross-phase references
    phase3 = """
class Experiment:
    def __init__(self, name, processor):
        self.name = name
        self.processor = processor

    def run(self):
        return {
            'experiment': self.name,
            'processor_name': self.processor.name,
            'stats': self.processor.stats(),
        }

exp1 = Experiment("exp_alpha", processor1)
exp1_result = exp1.run()

def summarize(proc):
    s = proc.stats()
    return s['mean'] + s['max']

summary1 = summarize(processor1)
summary2 = summarize(processor2)
"""

    execute_sandboxed(phase3, agent, state)
    state.commit()

    exp1_result = state.get("exp1_result")
    assert exp1_result["experiment"] == "exp_alpha"
    assert exp1_result["processor_name"] == "dataset1"
    assert exp1_result["stats"]["sum"] == 30
    assert state.get("summary1") == 6.0 + 10.0  # mean=6, max=10
    assert state.get("summary2") == 40.0 + 60.0  # mean=40, max=60

    # Phase 4: Large data and memory stress
    phase4 = """
large_1d = np.arange(1000)
large_2d = np.arange(5000).reshape(50, 100)
large_3d = np.zeros((10, 10, 10))

row_sums = np.sum(large_2d, axis=1)
col_sums = np.sum(large_2d, axis=0)

def matrix_stats(matrix):
    return {
        'shape': matrix.shape,
        'total': float(np.sum(matrix)),
        'mean': float(np.mean(matrix)),
    }

matrix_info = matrix_stats(large_2d)
total_elements = large_1d.shape[0] + large_2d.size + large_3d.size
"""

    execute_sandboxed(phase4, agent, state)
    state.commit()

    assert state.get("large_2d").shape == (50, 100)
    assert state.get("matrix_info")["shape"] == (50, 100)
    assert state.get("total_elements") == 1000 + 5000 + 1000

    # Phase 5: Cross-phase integration
    phase5 = """
# Use functions and objects from ALL previous phases
integration = {}

# Phase 1 functions
integration['multiply_test'] = multiply(7, 3)
integration['array_sum'] = int(np.sum(arr1))

# Phase 2 instances
integration['proc1_stats'] = processor1.stats()
integration['proc2_stats'] = processor2.stats()

# Phase 3 functions
integration['summary_test'] = summarize(processor1)

# Phase 3 nested instance
integration['exp_result'] = exp1.run()

# Phase 4 large data
integration['matrix_info'] = matrix_stats(large_2d)
integration['row_sum_total'] = float(np.sum(row_sums))

# New computation combining everything
grand_total = (
    integration['multiply_test']
    + integration['proc1_stats']['sum']
    + integration['proc2_stats']['sum']
    + int(integration['row_sum_total'])
)
"""

    execute_sandboxed(phase5, agent, state)
    state.commit()

    integration = state.get("integration")
    assert integration["multiply_test"] == 21
    assert integration["proc1_stats"]["sum"] == 30
    assert integration["proc2_stats"]["sum"] == 120
    assert integration["exp_result"]["experiment"] == "exp_alpha"
    assert integration["matrix_info"]["shape"] == (50, 100)

    grand_total = state.get("grand_total")
    # 21 + 30 + 120 + sum(arange(5000)) = 21 + 30 + 120 + 12497500
    expected = 21 + 30 + 120 + int(np.sum(np.arange(5000)))
    assert grand_total == expected

    # Rehydration cycles: load state into fresh agents and verify everything works
    for cycle in range(3):
        agent_new = Agent()
        agent_new.module(np, name="np")
        state_new = _make_versioned(store, commit_hash=state.current_commit)

        rehydration_test = f"""
cycle_results = {{}}

# Phase 1 functions still work
cycle_results['multiply'] = multiply(10, {cycle + 1})
cycle_results['arr_len'] = len(arr1)

# Phase 2 instances still work
cycle_results['proc1_name'] = processor1.name
cycle_results['proc2_stats'] = processor2.stats()

# Phase 3 nested instance still works
cycle_results['exp_name'] = exp1.name

# Phase 3 function still works
cycle_results['summary'] = summarize(processor1)

# Phase 4 large data intact
cycle_results['large_shape'] = large_2d.shape
cycle_results['matrix_check'] = matrix_stats(large_2d)

# Phase 5 integration dict intact
cycle_results['integration_keys'] = sorted(integration.keys())
"""

        execute_sandboxed(rehydration_test, agent_new, state_new)

        cr = state_new.get("cycle_results")
        assert cr["multiply"] == 10 * (cycle + 1)
        assert cr["arr_len"] == 100
        assert cr["proc1_name"] == "dataset1"
        assert cr["proc2_stats"]["sum"] == 120
        assert cr["exp_name"] == "exp_alpha"
        assert cr["summary"] == 16.0  # mean=6 + max=10
        assert cr["large_shape"] == (50, 100)
        assert cr["matrix_check"]["shape"] == (50, 100)
        assert len(cr["integration_keys"]) > 0
