"""Test that closure mutations work correctly with live state."""

import pytest

from agex import (
    Agent,
    clear_agent_registry,
    connect_fs,
    connect_state,
    run_file_in_sandbox,
)


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestClosureMutation:
    """Test that dict mutations in closures persist."""

    def test_dict_mutation_in_callback(self):
        """Dict mutations in nested function calls should persist."""
        agent = Agent(
            name="test",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        # Write test code that mutates a dict in a closure
        code = """
state = {"count": 0}
results = []

def increment():
    state["count"] += 1
    results.append(state["count"])

def save_state():
    results.append(f"saved:{state['count']}")

def do_update():
    increment()
    save_state()

# Call it multiple times
do_update()
do_update()
do_update()
"""
        fs = agent.fs("test")
        fs.write("test.py", code.encode())

        run_file_in_sandbox(agent, "test.py", "test")

        state = agent.state("test")
        results = state.get("results")

        print(f"results: {results}")
        print(f"state dict: {state.get('state')}")

        # Should be [1, "saved:1", 2, "saved:2", 3, "saved:3"]
        assert results == [1, "saved:1", 2, "saved:2", 3, "saved:3"], f"Got: {results}"

    def test_callback_style_execution(self):
        """Simulate NiceGUI-style callback: function defined in one run, called later."""
        agent = Agent(
            name="test_callback",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        # First run: define state and functions
        setup_code = """
state = {"count": 0}

def increment():
    state["count"] += 1

def get_count():
    return state["count"]
"""
        fs = agent.fs("test")
        fs.write("setup.py", setup_code.encode())
        run_file_in_sandbox(agent, "setup.py", "test")

        # Get the functions from state
        state_obj = agent.state("test")
        increment = state_obj.get("increment")
        get_count = state_obj.get("get_count")

        print(f"Before: count = {get_count()}")

        # Call increment multiple times (simulating button clicks)
        increment()
        increment()
        increment()

        print(f"After: count = {get_count()}")

        # Check the state dict directly
        state_dict = state_obj.get("state")
        print(f"state dict: {state_dict}")

        assert get_count() == 3, f"Expected 3, got {get_count()}"
        assert (
            state_dict["count"] == 3
        ), f"Expected state dict count=3, got {state_dict}"

    def test_file_write_with_closure(self):
        """Test that file writes capture current closure state, not initial."""
        import json

        agent = Agent(
            name="test_file",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )
        agent.module(json)

        code = """
import json

state = {"count": 0}

def save_state():
    with open('debug/state.json', 'w') as f:
        json.dump(state, f)

def increment_and_save():
    state["count"] += 1
    save_state()

# Initial save
save_state()

# Increment a few times
increment_and_save()
increment_and_save()
increment_and_save()
"""
        fs = agent.fs("test")
        fs.write("test.py", code.encode())
        run_file_in_sandbox(agent, "test.py", "test")

        # Read the file that was written
        content = fs.read("debug/state.json").decode()
        saved_state = json.loads(content)
        print(f"saved_state from file: {saved_state}")

        assert saved_state["count"] == 3, f"Expected count=3, got {saved_state}"

    def test_dict_mutation_simple(self):
        """Simple dict mutation should work."""
        agent = Agent(
            name="test2",
            state=connect_state(type="live", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        code = """
data = {"value": 0}

def modify():
    data["value"] = 42

modify()
result = data["value"]
"""
        fs = agent.fs("test")
        fs.write("test.py", code.encode())

        run_file_in_sandbox(agent, "test.py", "test")

        state = agent.state("test")
        result = state.get("result")

        print(f"result: {result}")
        assert result == 42, f"Got: {result}"
