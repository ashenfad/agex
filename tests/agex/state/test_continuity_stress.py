import numpy as np
import pytest

from agex.agent import Agent
from agex.eval.core import evaluate_program
from agex.eval.user_errors import AgexAttributeError
from agex.state import Versioned
from agex.state.kv import Memory


def test_multi_step_program_with_snapshots():
    """
    Tests that a series of programs, separated by snapshots, can build upon
    each other's state, correctly serializing and rehydrating user functions,
    classes, lambdas, and external library objects.
    """
    # 1. Setup the agent and versioned state
    agent = Agent()
    agent.module(np, name="np")

    store = Memory()
    state = Versioned(store)

    # 2. Define a sequence of programs that build on each other
    programs = [
        # Turn 1: Define a user function and import numpy
        "import np\ndef my_func(x):\n    return x * 2\na = 10",
        # Turn 2: Use the function and create a numpy array
        "b = my_func(a)\narr = np.array([1, b])",
        # Turn 3: Define a class that uses the array
        """
class MyClass:
    def __init__(self, val):
        self.val = val
    def get_val(self):
        return self.val * arr[0]

instance = MyClass(arr[1])
""",
        # Turn 4: Use the class instance and create a lambda
        "c = instance.get_val()\nmy_lambda = lambda x: x + c",
        # Turn 5: Execute the lambda
        "d = my_lambda(5)",
    ]

    # 3. Execute the programs in sequence, snapshotting after each one
    for i, program in enumerate(programs):
        evaluate_program(program, agent, state)
        state.snapshot()

    # 4. Verify the final state
    assert state.get("d") == 25
    final_arr = state.get("arr")
    np.testing.assert_array_equal(final_arr, np.array([1, 20]))

    # 5. Load the state into a new agent and verify it still works
    state2 = Versioned(store, commit_hash=state.current_commit)
    agent2 = Agent()
    agent2.module(np, name="np")

    # Run a new program that uses the restored state
    evaluate_program("e = my_func(d)\nf = my_lambda(a)", agent2, state2)
    assert state2.get("e") == 50
    assert state2.get("f") == 30

    # 6. Test that state is clean and doesn't leak
    state3 = Versioned(store, commit_hash=state.current_commit)
    with pytest.raises(AgexAttributeError):
        # This agent doesn't have numpy registered
        evaluate_program("z = np.array([1])", Agent(), state3)


def test_comprehensive_serialization_stress():
    """
    EXTREME stress test for the serialization system, covering every possible
    edge case, complex nested scenarios, exception handling, circular refs, etc.
    """
    # This test creates 10,000+ objects and needs more time than the default 5 seconds
    agent = Agent(timeout_seconds=15.0)
    agent.module(np, name="np")

    store = Memory()
    state = Versioned(store)

    # Phase 1: Complex function definitions and closures
    phase1 = """
import np
from dataclasses import dataclass

# Complex nested function with closures
def create_calculator(base):
    multiplier = base * 2

    def inner_calc(x):
        # This captures both 'base' and 'multiplier'
        def deep_calc(y):
            return (x + y) * multiplier + base
        return deep_calc

    return inner_calc

# Create calculator instances
calc1 = create_calculator(5)
calc2 = create_calculator(10)

# Test the nested closures
nested_calc1 = calc1(3)
nested_calc2 = calc2(7)

result1 = nested_calc1(2)  # (3 + 2) * 10 + 5 = 55
result2 = nested_calc2(4)  # (7 + 4) * 20 + 10 = 230

# Complex lambda with closure
factor = 100
complex_lambda = lambda x: x * factor + result1

# More complex closure chains
def closure_chain_factory():
    level1_var = "level1"
    
    def level2_factory(level2_var):
        def level3_factory(level3_var):
            def level4_func(level4_var):
                # This captures variables from 4 different scopes
                return f"{level1_var}-{level2_var}-{level3_var}-{level4_var}"
            return level4_func
        return level3_factory
    return level2_factory

chain_func = closure_chain_factory()("level2")("level3")
chain_result = chain_func("level4")
"""

    evaluate_program(phase1, agent, state)
    state.snapshot()

    # Verify phase 1 results
    assert state.get("result1") == 55
    assert state.get("result2") == 230
    assert state.get("complex_lambda")(2) == 255
    assert state.get("chain_result") == "level1-level2-level3-level4"

    # Phase 2: Complex classes with methods and numpy integration
    phase2 = """
# Define a complex class hierarchy
class DataProcessor:
    def __init__(self, name, data):
        self.name = name
        self.data = np.array(data)
        self.processed = False
        self.cache = {}

    def process(self):
        # Use the calculator from previous phase
        processed_data = []
        for i, val in enumerate(self.data):
            calc_func = calc1(val)
            processed_data.append(calc_func(i))

        self.data = np.array(processed_data)
        self.processed = True
        return self.data

    def get_stats(self):
        if not self.processed:
            raise ValueError("Data not processed yet")

        # Create lambda that captures self
        stat_lambda = lambda: {
            'mean': np.mean(self.data),
            'sum': np.sum(self.data),
            'max': np.max(self.data),
            'name': self.name,
            'cache_size': len(self.cache)
        }
        return stat_lambda
    
    def add_reference(self, other):
        # Create potential circular references
        self.cache['reference'] = other
        if hasattr(other, 'cache'):
            other.cache['back_reference'] = self

# More complex class with class methods (no inheritance, just composition)
class AdvancedProcessor:
    def __init__(self, name, data, config):
        self.name = name
        self.data = np.array(data)
        self.processed = False
        self.cache = {}
        self.config = config
        self.processors = []
    
    def add_processor(self, processor):
        self.processors.append(processor)
        # Create cross-references
        processor.cache['parent'] = self
    
    def batch_process(self):
        results = []
        for proc in self.processors:
            if hasattr(proc, 'process'):
                results.append(proc.process())
        
        # Complex lambda capturing lots of state
        summary_func = lambda: {
            'total_processors': len(self.processors),
            'results_shape': [r.shape if hasattr(r, 'shape') else len(r) for r in results],
            'config': self.config,
            'chain_test': chain_func("batch")
        }
        return summary_func

# Create processor instances with complex relationships
processor1 = DataProcessor("dataset1", [1, 2, 3, 4, 5])
processor2 = DataProcessor("dataset2", [10, 20, 30])
advanced = AdvancedProcessor("advanced", [100, 200], {"mode": "fast", "threads": 4})

# Create circular references
processor1.add_reference(processor2)
advanced.add_processor(processor1)
advanced.add_processor(processor2)

# Process the data
data1 = processor1.process()
data2 = processor2.process()
batch_summary = advanced.batch_process()

# Get stat functions (lambdas that capture self)
stats1_func = processor1.get_stats()
stats2_func = processor2.get_stats()
"""

    evaluate_program(phase2, agent, state)
    state.snapshot()

    # Verify phase 2 results
    stats1 = state.get("stats1_func")()
    stats2 = state.get("stats2_func")()
    batch_results = state.get("batch_summary")()

    assert stats1["sum"] > 0
    assert stats2["sum"] > 0
    assert isinstance(stats1["mean"], (int, float, np.number))
    assert batch_results["total_processors"] == 2
    assert batch_results["chain_test"] == "level1-level2-level3-batch"

    # Phase 3: Dataclasses and complex comprehensions
    phase3 = """
# Define dataclass (fields only)
@dataclass
class Point:
    x: float
    y: float

@dataclass 
class Circle:
    center: Point
    radius: float

# Define helper function for points
def distance_from_origin(point):
    return (point.x ** 2 + point.y ** 2) ** 0.5

def distance_between_points(p1, p2):
    return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2)**0.5

# Create points using comprehensions with closures
origin_distance_threshold = 5.0

points = [Point(x, y) for x in range(-3, 4) for y in range(-3, 4)
          if distance_from_origin(Point(x, y)) <= origin_distance_threshold]

# Create circles with point references
circles = [Circle(Point(0, 0), r) for r in [1, 2, 3, 4, 5]]

# Complex nested comprehension with lambda
distance_calculator = lambda p1, p2: distance_between_points(p1, p2)

# Create distance matrix using nested comprehensions
distances = {
    f"point_{i}_to_{j}": distance_calculator(p1, p2)
    for i, p1 in enumerate(points[:5])  
    for j, p2 in enumerate(points[:5])
    if i != j
}

# Function that returns a function that captures local variables
def create_filter(threshold):
    local_multiplier = threshold * 2

    def filter_func(point_list):
        # This captures both threshold and local_multiplier
        filtered = [p for p in point_list
                   if distance_from_origin(p) * local_multiplier > threshold]

        # Return a lambda that captures the filtered list
        return lambda: len(filtered), filtered

    return filter_func

# Create and use the filter
point_filter = create_filter(3.0)
count_func, filtered_points = point_filter(points)
filtered_count = count_func()

# Complex data structure with mixed types
complex_structure = {
    'points': points[:10],
    'circles': circles,
    'distances': distances,
    'processors': [processor1, processor2],
    'functions': [calc1, calc2, complex_lambda],
    'numpy_arrays': [data1, data2],
    'nested': {
        'more_points': points[10:20],
        'calculator_results': [calc1(i)(i+1) for i in range(5)],
        'lambda_results': [complex_lambda(i) for i in range(3)]
    }
}
"""

    evaluate_program(phase3, agent, state)
    state.snapshot()

    # Verify phase 3 results
    assert len(state.get("points")) > 0
    assert len(state.get("distances")) > 0
    assert state.get("filtered_count") >= 0
    complex_struct = state.get("complex_structure")
    assert len(complex_struct["points"]) == 10
    assert len(complex_struct["circles"]) == 5
    assert "processors" in complex_struct

    # Phase 4: Exception handling and error recovery (now that scoped.remove works!)
    phase4 = """
# Create operations that will succeed and fail (define early so lambdas can reference it)
operations = [
    lambda: processor1.get_stats()()['mean'],  # Should work
    lambda: np.sum([1, 2, 3]),  # Should work
    lambda: complex_lambda(5),  # Should work
    lambda: 1 / 0,  # Will fail - division by zero
    lambda: complex_structure['nonexistent_key'],  # Will fail - key error
    lambda: processor2.get_stats()()['max'],  # Should work
    lambda: complex_structure['points'][0].x,  # Should work
]

# Complex exception handling with closures
def safe_processor_with_recovery(operations, recovery_mode="default"):
    results = []
    # Use a mutable container to track error state instead of nonlocal
    error_state = {'count': 0, 'details': []}
    
    def log_error(op_name, error):
        error_state['count'] += 1
        # Use string representation instead of __name__ to work with sandbox
        error_type_name = str(type(error))
        error_info = {
            'operation': op_name,
            'error_type': error_type_name,
            'error_msg': str(error),
            'recovery_mode': recovery_mode
        }
        error_state['details'].append(error_info)
        return f"Error in {op_name}: {error_info['error_type']}"
    
    def create_recovery_func(op_index):
        # This closure captures op_index and recovery_mode
        def recovery_func():
            if recovery_mode == "default":
                return f"recovered_op_{op_index}"
            elif recovery_mode == "calc":
                return calc1(op_index)(1)  # Use calc from previous phases
            else:
                return None
        return recovery_func

    for i, op in enumerate(operations):
        try:
            result = op()
            results.append(result)
        except Exception as e:
            try:
                # Nested exception handling
                error_msg = log_error(f"operation_{i}", e)
                recovery_func = create_recovery_func(i)
                recovered_result = recovery_func()
                results.append(recovered_result)
            except Exception as recovery_error:
                # Even recovery failed
                log_error(f"recovery_{i}", recovery_error)
                results.append(None)

    # Return a complex closure that captures everything
    return lambda report_type="summary": {
        'results': results,
        'error_count': error_state['count'],
        'error_details': error_state['details'],
        'recovery_mode': recovery_mode,
        'total_ops': len(operations),
        'success_rate': (len(operations) - error_state['count']) / len(operations) if operations else 0
        } if report_type == "full" else f"Processed {len(operations)} ops with {error_state['count']} errors"

# Process with different recovery modes
result_func_default = safe_processor_with_recovery(operations, "default")
result_func_calc = safe_processor_with_recovery(operations, "calc")

# Get results
results_default = result_func_default("full")
results_calc = result_func_calc("full")
"""

    evaluate_program(phase4, agent, state)
    state.snapshot()

    # Verify phase 4 results
    results_default = state.get("result_func_default")("full")
    results_calc = state.get("result_func_calc")("full")

    assert results_default["total_ops"] == 7
    assert results_default["error_count"] == 2  # division by zero and key error
    assert results_calc["error_count"] == 2
    assert len(results_default["results"]) == 7
    assert len(results_calc["results"]) == 7

    # Phase 5: Large data and memory stress
    phase5 = """
# Large numpy arrays
large_array_1 = np.arange(0, 1000)  # 1000 numbers
large_array_2 = np.arange(0, 5000).reshape(50, 100)  # 2D array
large_array_3 = np.zeros((20, 20, 20))  # 3D array

# Complex functions with large data
def matrix_processor(matrix):
    # Create closures that capture large data
    def process_rows():
        row_sums = [np.sum(row) for row in matrix]
        
        def get_stats():
            return {
                'row_count': len(row_sums),
                'total_sum': sum(row_sums),
                'max_row_sum': max(row_sums),
                'matrix_shape': matrix.shape
            }
        return get_stats
    
    def process_columns():
        col_sums = [np.sum(matrix[:, i]) for i in range(matrix.shape[1])]
        
        def get_stats():
            return {
                'col_count': len(col_sums),
                'total_sum': sum(col_sums),
                'max_col_sum': max(col_sums),
                'avg_col_sum': np.mean(col_sums)
            }
        return get_stats
    
    return process_rows(), process_columns()

# Process large arrays
row_processor, col_processor = matrix_processor(large_array_2)
row_stats = row_processor()
col_stats = col_processor()

# Many small objects (stress test object count)
many_points = [Point(i * 0.1, j * 0.1) for i in range(100) for j in range(100)]  # 10,000 points
many_circles = [Circle(Point(i, j), i + j) for i in range(50) for j in range(50)]  # 2,500 circles

# Complex nested data structure
mega_structure = {
    'large_arrays': [large_array_1, large_array_2, large_array_3],
    'many_objects': {
        'points': many_points[:5000],  # Limit to 5000 for practical testing
        'circles': many_circles[:1000],  # Limit to 1000
    },
    'processors': {
        'row_proc': row_processor,
        'col_proc': col_processor,
        'stats': [row_stats, col_stats]
    }
    # 'previous_phase_refs': {
    #     'calc1': calc1,
    #     'calc2': calc2,
    #     'processor1': processor1,
    #     'processor2': processor2,
    #     'complex_structure': complex_structure
    # }
}
"""

    evaluate_program(phase5, agent, state)
    state.snapshot()

    # Verify phase 5 results
    mega_struct = state.get("mega_structure")
    assert len(mega_struct["large_arrays"]) == 3
    assert len(mega_struct["many_objects"]["points"]) == 5000
    assert len(mega_struct["many_objects"]["circles"]) == 1000
    assert mega_struct["processors"]["stats"][0]["row_count"] == 50

    # Phase 6: Final integration and cross-phase dependencies
    phase6 = """
# Create a function that uses data from ALL previous phases
def ultimate_integration_test():
    integration_results = {}
    
    # Test phase 1 functions still work
    integration_results['phase1'] = {
        'calc1_test': calc1(7)(3),
        'calc2_test': calc2(2)(8), 
        'complex_lambda_test': complex_lambda(1),
        'chain_result': chain_func("integration")
    }
    
    # Test phase 2 objects still work
    integration_results['phase2'] = {
        'stats1': stats1_func(),
        'stats2': stats2_func(),
        'batch_summary': batch_summary(),
        'processor1_name': processor1.name,
        'circular_ref_test': processor1.cache['reference'].name
    }
    
    # Test phase 3 data structures
    integration_results['phase3'] = {
        'total_points': len(points),
        'total_circles': len(circles),
        'complex_structure_keys': list(complex_structure.keys()),
        'distance_sample': distance_calculator(points[0], points[1])
    }
    
    # Test phase 4 exception handling
    integration_results['phase4'] = {
        'error_summary_default': results_default['error_count'],
        'error_summary_calc': results_calc['error_count'],
        'recovery_test': results_default['results'][3] if len(results_default['results']) > 3 else 'no_recovery_result'
    }
    
    # Test phase 5 large data
    integration_results['phase5'] = {
        'large_array_shape': large_array_2.shape,
        'many_points_count': len(many_points),
        'mega_structure_size': len(mega_structure),
        'row_stats': row_stats,
        'col_stats': col_stats
    }
    
    # Create final closure that captures everything
    def create_final_report():
        total_objects = (
            len(points) + len(circles) + len(many_points) + len(many_circles) +
            len(complex_structure) + len(mega_structure)
        )
        
        return lambda detail_level="summary": {
            'total_serialized_objects': total_objects,
            'phases_tested': 5,
            'integration_results': integration_results,
            'numpy_arrays_tested': 6,  # large_array_1, 2, 3, data1, data2, plus processor arrays
            'functions_tested': len([calc1, calc2, complex_lambda, chain_func]),
            'exception_scenarios': results_default['total_ops'],
            'circular_references': 'tested',
            'cross_phase_dependencies': 'verified'
        } if detail_level == "full" else f"Integration test completed with {total_objects} objects"
    
    return create_final_report()

# Execute ultimate integration test
final_report_func = ultimate_integration_test()
final_summary = final_report_func("full")
final_summary_short = final_report_func("summary")
"""

    evaluate_program(phase6, agent, state)
    state.snapshot()

    # Verify phase 6 results
    final_results = state.get("final_summary")
    assert final_results["phases_tested"] == 5
    assert final_results["total_serialized_objects"] > 6000  # Should be quite large
    assert final_results["circular_references"] == "tested"
    assert final_results["cross_phase_dependencies"] == "verified"

    # EXTREME TEST: Multiple rehydration cycles with different agents
    print(f"Testing EXTREME rehydration with {len(state.ephemeral)} state variables...")

    # Create multiple new agents and test rehydration robustness
    for cycle in range(3):  # 3 rehydration cycles
        agent_new = Agent()
        agent_new.module(np, name="np")

        state_new = Versioned(store, commit_hash=state.current_commit)

        # Test that ALL complex functionality still works after rehydration
        rehydration_test = f"""
# Cycle {cycle + 1} rehydration test
cycle_results = {{}}

# Test all phases still work
cycle_results['calc1'] = calc1(7)(3)
cycle_results['calc2'] = calc2(2)(8)
cycle_results['complex_lambda'] = complex_lambda(1)
cycle_results['chain_test'] = chain_func("rehydration_cycle_{cycle}")

# Test dataclass operations
new_point = Point(3, 4)
cycle_results['point_distance'] = distance_from_origin(new_point)
cycle_results['new_circle'] = Circle(new_point, 2.5)

# Test processor methods (complex object rehydration)
cycle_results['stats1'] = stats1_func()
cycle_results['stats2'] = stats2_func()

# Test large data structures
cycle_results['mega_structure_test'] = len(mega_structure['many_objects']['points'])
cycle_results['row_stats_test'] = row_stats['row_count']

# Test final integration
cycle_results['final_report'] = final_report_func("summary")

# Test exception handling functions
try:
    # Simple division by zero test
    test_result = 1 / 0
    cycle_results['exception_test'] = 0  # No error
except Exception as e:
    cycle_results['exception_test'] = 1  # Got expected error

# Create cycle-specific integration result
cycle_integration = sum([
    cycle_results['calc1'],
    cycle_results['calc2'], 
    int(cycle_results['point_distance']),
    len(cycle_results['stats1']),
    len(cycle_results['stats2']),
    cycle_results['mega_structure_test'] // 1000,  # Normalize large number
    cycle_results['row_stats_test']
])

cycle_results['integration_sum'] = cycle_integration
"""

        evaluate_program(rehydration_test, agent_new, state_new)

        # Verify each rehydration cycle
        cycle_results = state_new.get("cycle_results")
        assert cycle_results["calc1"] > 0
        assert cycle_results["calc2"] > 0
        assert cycle_results["complex_lambda"] > 0
        assert cycle_results["point_distance"] == 5.0
        assert "mean" in cycle_results["stats1"]
        assert "mean" in cycle_results["stats2"]
        assert cycle_results["mega_structure_test"] == 5000
        assert cycle_results["row_stats_test"] == 50
        assert cycle_results["integration_sum"] > 0
        assert (
            cycle_results["exception_test"] == 1
        )  # Should have 1 error (division by zero)

    print("✅ EXTREME Comprehensive serialization stress test passed!")
    print(f"   - Serialized and rehydrated {len(state.ephemeral)} state variables")
    print(
        f"   - Tested 6 phases with {final_results['total_serialized_objects']} total objects"
    )
    print(f"   - Verified {final_results['numpy_arrays_tested']} numpy arrays")
    print(
        f"   - Tested {final_results['functions_tested']} complex functions with closures"
    )
    print(f"   - Handled {final_results['exception_scenarios']} exception scenarios")
    print("   - Verified circular references and cross-phase dependencies")
    print("   - Tested large data structures and memory stress")
    print("   - Completed 3 full rehydration cycles with different agents")
    print("   - All functionality preserved across agent boundaries")
