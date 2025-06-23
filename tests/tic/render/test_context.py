from tic.render.context import ContextRenderer
from tic.state.kv import Memory
from tic.state.versioned import Versioned


def test_render_separate_streams():
    """Checks that state and stdout are rendered into separate blocks."""
    state = Versioned(Memory())
    state.set("x", 1)
    state.set("__stdout__", ["Hello"])
    state.set("y", 2)

    renderer = ContextRenderer("gpt-4o")
    # High budget, everything should be visible.
    output = renderer.render(state, budget=100)

    assert "x = 1\ny = 2" in output
    assert "Agent printed:\n'Hello'" in output


def test_state_budget_truncation():
    """Checks that the state stream is truncated independently."""
    state = Versioned(Memory())
    state.set("x", "a" * 100)  # This will be truncated
    state.set("y", 2)  # This should be visible
    state.set("__stdout__", ["Hello"])

    renderer = ContextRenderer("gpt-4o")
    # Budget for state is 60% of 20 = 12 tokens.
    # Should only be enough for y=2.
    output = renderer.render(state, budget=20)

    assert "y = 2" in output
    assert "x =" not in output
    # stdout should be unaffected
    assert "Agent printed:\n'Hello'" in output


def test_stdout_budget_truncation():
    """Checks that the stdout stream is truncated independently."""
    state = Versioned(Memory())
    state.set("x", 1)
    state.set(
        "__stdout__",
        [
            "This is a very long line that will be truncated",
            "This should be visible",
        ],
    )

    renderer = ContextRenderer("gpt-4o")
    # Budget for stdout is 40% of 40 = 16 tokens.
    # Not enough for the long line, but enough for the short one.
    output = renderer.render(state, budget=45)

    # State should be unaffected
    assert "x = 1" in output
    assert "Agent printed:" in output
    assert "..." in output
    assert "'This should be visible'" in output
    assert "very long line" not in output


def test_large_collection_in_state():
    """Checks that large collections in state are summarized."""
    state = Versioned(Memory())
    state.set("x", list(range(10000)))
    state.set("__stdout__", ["Hello"])

    renderer = ContextRenderer("gpt-4o")
    output = renderer.render(state, budget=100)

    assert "x = [... (10000 items)]" in output
    assert "Agent printed:\n'Hello'" in output


def test_large_collection_in_stdout():
    """Checks that large collections in stdout are summarized."""
    state = Versioned(Memory())
    state.set("x", 1)
    state.set("__stdout__", [list(range(10000)), "visible"])

    renderer = ContextRenderer("gpt-4o")
    output = renderer.render(state, budget=100)

    assert "x = 1" in output
    assert "Agent printed:" in output
    assert "[... (10000 items)]" in output
    assert "'visible'" in output
