from tic.render.context import ContextRenderer
from tic.state import kv
from tic.state.versioned import Versioned


def test_render_combined_streams():
    """
    Checks that ContextRenderer correctly combines the state and stdout streams.
    """
    state = Versioned(kv.Memory())
    state.set("x", 1)
    state.set("__stdout__", ["Hello"])
    state.set("y", 2)
    state.snapshot()

    renderer = ContextRenderer("gpt-4o")
    # High budget, everything should be visible.
    output = renderer.render(state, budget=100)

    assert "x = 1" in output
    assert "y = 2" in output
    assert "Agent printed:\n" in output
    assert "Hello" in output
