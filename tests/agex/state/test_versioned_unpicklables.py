"""Tests for unpicklable object handling in Versioned state."""

import pickle

import pytest
from kvgit import Staged, VersionedKV

from agex.agent.datatypes import UnpicklableMarker, UnpicklableVariableError
from agex.state import _agex_decoder, _agex_encoder, safe_commit
from agex.state.kv import Memory


def _make_versioned():
    return Staged(VersionedKV(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


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
    state = _make_versioned()

    # Create and use an unpicklable object in one turn
    unpicklable = UnpicklableObject(42)
    state["obj"] = unpicklable
    result = state.get("obj").process()

    assert result == 43

    # Checkpoint - should succeed and create a marker
    commit_result = state.commit()
    assert commit_result.merged
    assert commit_result.commit is not None


def test_multi_turn_unpicklable_raises_clear_error():
    """Attempting to access unpicklable variable in next turn should raise helpful error."""
    state = _make_versioned()

    # Turn 1: Create unpicklable object
    unpicklable = UnpicklableObject(42)
    state["cursor"] = unpicklable
    state.commit()

    # Turn 2: Try to access it — should raise UnpicklableVariableError
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("cursor")

    error_msg = str(exc_info.value)
    assert "UnpicklableObject" in error_msg


def test_safe_commit_skips_unpicklable_refs():
    """safe_commit should not raise when referenced_keys includes an UnpicklableMarker.

    Reproduces the bug where turn 1 creates an unpicklable variable (added to
    accumulated_refs via find_refs), and turn 2's safe_commit re-stages the
    key, triggering UnpicklableVariableError outside the sandbox.
    """
    state = _make_versioned()

    # Turn 1: create unpicklable and commit (marker stored in state)
    state["at"] = UnpicklableObject(42)
    state["good"] = "hello"
    state.commit()

    # Turn 2: only touch a different key, but accumulated_refs still has "at"
    state["good"] = "world"
    result = safe_commit(state, referenced_keys={"at", "good"})
    assert result.merged

    # "good" should reflect the new value
    assert state.get("good") == "world"

    # "at" should still be an unpicklable marker (not lost, not raised)
    with pytest.raises(UnpicklableVariableError):
        state.get("at")


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
    state = _make_versioned()

    # Create a list with an unpicklable element
    unpicklable = UnpicklableObject(42)
    mixed_list = [1, 2, unpicklable, 3]
    state["mixed_list"] = mixed_list
    state.commit()

    # Try to access the list
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("mixed_list")

    error_msg = str(exc_info.value)
    assert "list" in error_msg


def test_closure_over_unpicklable_marks_function():
    """Function closing over unpicklable should be marked as unpicklable."""
    state = _make_versioned()

    # Create closure over unpicklable object
    unpicklable = UnpicklableObject(42)

    def get_data():
        return unpicklable.process()

    state["cursor"] = unpicklable
    state["get_data"] = get_data
    state.commit()

    # Both should be marked as unpicklable
    with pytest.raises(UnpicklableVariableError):
        state.get("cursor")

    with pytest.raises(UnpicklableVariableError):
        state.get("get_data")


def test_mutation_to_unpicklable_creates_marker():
    """Object that becomes unpicklable after explicit re-set should get marked."""
    state = _make_versioned()

    # Start with picklable list
    my_list = [1, 2, 3]
    state["my_list"] = my_list
    state.commit()

    # Mutate and explicitly re-set (kvgit requires explicit set for changes)
    my_list.append(UnpicklableObject(42))
    state["my_list"] = my_list

    # Checkpoint should create a marker for the unpicklable value
    commit_result = state.commit()
    assert commit_result.merged

    # Next access should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("my_list")


def test_multiple_unpicklables_in_same_checkpoint():
    """Multiple unpicklable variables should all get markers."""
    state = _make_versioned()

    # Create multiple unpicklable objects
    state["cursor1"] = UnpicklableObject(1)
    state["cursor2"] = UnpicklableObject(2)
    state["file_handle"] = UnpicklableObject(3)
    state["picklable_data"] = [1, 2, 3]  # This one is fine

    commit_result = state.commit()
    assert commit_result.merged
    assert commit_result.commit is not None

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
    state = _make_versioned()

    # Turn 1: Create unpicklable
    state["cursor"] = UnpicklableObject(42)
    state.commit()

    # Turn 2: Create some other data
    state["data"] = [1, 2, 3]
    state.commit()

    # Turn 3: Create more data
    state["more_data"] = {"key": "value"}
    state.commit()

    # Cursor should still be unavailable
    with pytest.raises(UnpicklableVariableError):
        state.get("cursor")

    # But other data should be accessible
    assert state.get("data") == [1, 2, 3]
    assert state.get("more_data") == {"key": "value"}


def test_picklable_data_works_normally():
    """Normal picklable objects should continue to work without any changes."""
    state = _make_versioned()

    # Set various picklable types
    state["number"] = 42
    state["string"] = "hello"
    state["list"] = [1, 2, 3]
    state["dict"] = {"key": "value"}
    state["tuple"] = (1, 2, 3)

    state.commit()

    # All should be accessible
    assert state.get("number") == 42
    assert state.get("string") == "hello"
    assert state.get("list") == [1, 2, 3]
    assert state.get("dict") == {"key": "value"}
    assert state.get("tuple") == (1, 2, 3)


def test_checkout_with_unpicklable_markers():
    """Checking out a commit with markers should preserve the markers."""
    state = _make_versioned()

    # Turn 1: Create mixed data
    state["good_data"] = [1, 2, 3]
    state.commit()
    commit1 = state.current_commit

    # Turn 2: Add unpicklable
    state["bad_data"] = UnpicklableObject(42)
    state.commit()
    commit2 = state.current_commit

    # Turn 3: Add more good data
    state["more_good"] = "hello"
    state.commit()

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
    state = _make_versioned()

    results = {"count": 42, "data": [1, 2, 3], "cursor": UnpicklableObject(100)}

    state["results"] = results
    state.commit()

    # Whole dict should be unavailable
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("results")

    assert "dict" in str(exc_info.value)


def test_contains_check_with_unpicklable():
    """__contains__ check should work for variables with markers."""
    state = _make_versioned()

    state["cursor"] = UnpicklableObject(42)
    state.commit()

    # Should report that the key exists
    assert "cursor" in state

    # But accessing it should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("cursor")


def test_keys_includes_unpicklable_variables():
    """keys() should include variables that have markers."""
    state = _make_versioned()

    state["good"] = 42
    state["bad"] = UnpicklableObject(100)
    state.commit()

    keys = list(state.keys())
    assert "good" in keys
    assert "bad" in keys

    # Good should be accessible
    assert state.get("good") == 42

    # Bad should raise
    with pytest.raises(UnpicklableVariableError):
        state.get("bad")


def test_namespaced_key_displays_correctly_in_error():
    """Accessing unpicklable variable through namespaced key should raise."""
    state = _make_versioned()

    # Simulate namespaced key (like what Namespaced state would create)
    cursor = UnpicklableObject(42)
    state["my_agent/cursor"] = cursor
    state.commit()

    # Should raise UnpicklableVariableError
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("my_agent/cursor")

    error_msg = str(exc_info.value)
    assert "UnpicklableObject" in error_msg


def test_namespace_hydration_places_marker_instead_of_dropping():
    """Unpicklable variables should appear as markers in the namespace,
    not be silently dropped (which would cause NameError)."""
    from unittest.mock import MagicMock

    from agex.eval.bridge.namespace import build_namespace

    state = _make_versioned()

    # Store a picklable and an unpicklable variable
    state["good"] = 42
    state["bad"] = UnpicklableObject(99)
    state.commit()

    # Build namespace as the agent loop would
    agent = MagicMock()
    agent.policy = MagicMock()
    agent.policy.namespaces = []
    namespace, pre_keys, _ = build_namespace(state, agent, "test_agent")

    # Picklable variable is in the namespace normally
    assert namespace["good"] == 42
    assert "good" in pre_keys

    # Unpicklable variable is in the namespace as a marker (not dropped)
    assert "bad" in namespace
    assert "bad" in pre_keys
    assert isinstance(namespace["bad"], UnpicklableMarker)

    # Accessing any attribute on the marker raises a descriptive error
    with pytest.raises(UnpicklableVariableError) as exc_info:
        namespace["bad"].data
    assert "bad" in str(exc_info.value)
    assert "Re-create it" in str(exc_info.value)
