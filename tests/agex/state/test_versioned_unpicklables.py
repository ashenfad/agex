"""Tests for unpicklable object handling in Versioned state."""

import pickle

import pytest

from agex.agent.datatypes import UnpicklableMarker, UnpicklableVariableError
from agex.state import Versioned


class UnpicklableObject:
    """A simple class that cannot be pickled."""

    def __init__(self, data):
        self.data = data
        # Add something that makes it unpicklable (like a lambda)
        self.func = lambda x: x + 1

    def process(self):
        return self.func(self.data)


def test_single_turn_unpicklable_silent_success():
    """Single-turn use of unpicklable object should work seamlessly with no warnings."""
    state = Versioned()

    # Create and use an unpicklable object in one turn
    unpicklable = UnpicklableObject(42)
    state.set("obj", unpicklable)
    result = state.get("obj").process()

    assert result == 43

    # Checkpoint - should succeed and create a marker
    snapshot_result = state.snapshot()
    assert snapshot_result.commit_hash is not None
    assert len(snapshot_result.unsaved_keys) == 0  # Marker was successfully created


def test_multi_turn_unpicklable_raises_clear_error():
    """Attempting to access unpicklable variable in next turn should raise helpful error."""
    state = Versioned()

    # Turn 1: Create unpicklable object
    unpicklable = UnpicklableObject(42)
    state.set("cursor", unpicklable)
    state.snapshot()

    # Turn 2: Try to access it
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("cursor")

    error_msg = str(exc_info.value)
    assert "cursor" in error_msg
    assert "UnpicklableObject" in error_msg
    assert "not available" in error_msg
    assert "Solutions:" in error_msg
    assert "Recreate it" in error_msg
    assert "Chain operations" in error_msg


def test_marker_is_picklable():
    """The marker itself must be picklable."""
    marker = UnpicklableMarker(
        variable_name="test", type_name="TestClass", original_exception="test exception"
    )

    # Should not raise
    serialized = pickle.dumps(marker)
    deserialized = pickle.loads(serialized)

    assert deserialized.variable_name == "test"
    assert deserialized.type_name == "TestClass"


def test_nested_unpicklable_marks_whole_structure():
    """List/dict containing unpicklable should have whole structure marked."""
    state = Versioned()

    # Create a list with an unpicklable element
    unpicklable = UnpicklableObject(42)
    mixed_list = [1, 2, unpicklable, 3]
    state.set("mixed_list", mixed_list)
    state.snapshot()

    # Try to access the list
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("mixed_list")

    error_msg = str(exc_info.value)
    assert "mixed_list" in error_msg
    assert "list" in error_msg


def test_closure_over_unpicklable_marks_function():
    """Function closing over unpicklable should be marked as unpicklable."""
    state = Versioned()

    # Create closure over unpicklable object
    unpicklable = UnpicklableObject(42)

    def get_data():
        return unpicklable.process()

    state.set("cursor", unpicklable)
    state.set("get_data", get_data)
    state.snapshot()

    # Both should be marked as unpicklable
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("cursor")
    assert "cursor" in str(exc_info.value)

    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("get_data")
    assert "get_data" in str(exc_info.value)
    assert "function" in str(exc_info.value)


def test_mutation_to_unpicklable_creates_marker():
    """Object that becomes unpicklable after mutation should get marked."""
    state = Versioned()

    # Start with picklable list
    my_list = [1, 2, 3]
    state.set("my_list", my_list)
    state.snapshot()

    # Get the list (triggers mutation tracking)
    retrieved_list = state.get("my_list")

    # Mutate it to be unpicklable
    retrieved_list.append(UnpicklableObject(42))

    # Checkpoint should detect the mutation and create a marker
    snapshot_result = state.snapshot()
    assert snapshot_result.commit_hash is not None

    # Next access should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("my_list")


def test_multiple_unpicklables_in_same_checkpoint():
    """Multiple unpicklable variables should all get markers."""
    state = Versioned()

    # Create multiple unpicklable objects
    state.set("cursor1", UnpicklableObject(1))
    state.set("cursor2", UnpicklableObject(2))
    state.set("file_handle", UnpicklableObject(3))
    state.set("picklable_data", [1, 2, 3])  # This one is fine

    snapshot_result = state.snapshot()
    assert snapshot_result.commit_hash is not None
    assert len(snapshot_result.unsaved_keys) == 0

    # Picklable data should work
    assert state.get("picklable_data") == [1, 2, 3]

    # Unpicklable ones should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("cursor1")

    with pytest.raises(UnpicklableVariableError):
        state.get("cursor2")

    with pytest.raises(UnpicklableVariableError):
        state.get("file_handle")


def test_marker_persists_across_multiple_checkpoints():
    """Marker should survive across multiple checkpoints."""
    state = Versioned()

    # Turn 1: Create unpicklable
    state.set("cursor", UnpicklableObject(42))
    state.snapshot()

    # Turn 2: Create some other data
    state.set("data", [1, 2, 3])
    state.snapshot()

    # Turn 3: Create more data
    state.set("more_data", {"key": "value"})
    state.snapshot()

    # Cursor should still be unavailable
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("cursor")
    assert "cursor" in str(exc_info.value)

    # But other data should be accessible
    assert state.get("data") == [1, 2, 3]
    assert state.get("more_data") == {"key": "value"}


def test_picklable_data_works_normally():
    """Normal picklable objects should continue to work without any changes."""
    state = Versioned()

    # Set various picklable types
    state.set("number", 42)
    state.set("string", "hello")
    state.set("list", [1, 2, 3])
    state.set("dict", {"key": "value"})
    state.set("tuple", (1, 2, 3))

    state.snapshot()

    # All should be accessible
    assert state.get("number") == 42
    assert state.get("string") == "hello"
    assert state.get("list") == [1, 2, 3]
    assert state.get("dict") == {"key": "value"}
    assert state.get("tuple") == (1, 2, 3)


def test_checkout_with_unpicklable_markers():
    """Checking out a commit with markers should preserve the markers."""
    state = Versioned()

    # Turn 1: Create mixed data
    state.set("good_data", [1, 2, 3])
    state.snapshot()
    commit1 = state.current_commit

    # Turn 2: Add unpicklable
    state.set("bad_data", UnpicklableObject(42))
    state.snapshot()
    commit2 = state.current_commit

    # Turn 3: Add more good data
    state.set("more_good", "hello")
    state.snapshot()

    # Checkout commit2 (has the marker)
    state2 = state.checkout(commit2)

    # Good data should be accessible
    assert state2.get("good_data") == [1, 2, 3]

    # Bad data should raise
    with pytest.raises(UnpicklableVariableError):
        state2.get("bad_data")

    # more_good shouldn't exist yet
    assert state2.get("more_good") is None

    # Checkout commit1 (before the unpicklable)
    state1 = state.checkout(commit1)
    assert state1.get("good_data") == [1, 2, 3]
    assert state1.get("bad_data") is None


def test_dict_with_unpicklable_values():
    """Dictionary with unpicklable values should be marked."""
    state = Versioned()

    results = {"count": 42, "data": [1, 2, 3], "cursor": UnpicklableObject(100)}

    state.set("results", results)
    state.snapshot()

    # Whole dict should be unavailable
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("results")

    assert "results" in str(exc_info.value)
    assert "dict" in str(exc_info.value)


def test_contains_check_with_unpicklable():
    """__contains__ check should work for variables with markers."""
    state = Versioned()

    state.set("cursor", UnpicklableObject(42))
    state.snapshot()

    # Should report that the key exists
    assert "cursor" in state

    # But accessing it should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("cursor")


def test_keys_includes_unpicklable_variables():
    """keys() should include variables that have markers."""
    state = Versioned()

    state.set("good", 42)
    state.set("bad", UnpicklableObject(100))
    state.snapshot()

    keys = list(state.keys())
    assert "good" in keys
    assert "bad" in keys

    # Good should be accessible
    assert state.get("good") == 42

    # Bad should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("bad")


def test_namespaced_key_displays_correctly_in_error():
    """Error message should show variable name without namespace prefix."""
    state = Versioned()

    # Simulate namespaced key (like what Namespaced state would create)
    cursor = UnpicklableObject(42)
    state.set("my_agent/cursor", cursor)
    state.snapshot()

    # Error message should show "cursor", not "my_agent/cursor"
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("my_agent/cursor")

    error_msg = str(exc_info.value)
    # Should show clean variable name
    assert "Variable 'cursor'" in error_msg
    assert "Recreate it: cursor = " in error_msg
    # Should NOT show namespace prefix
    assert "my_agent/cursor" not in error_msg
