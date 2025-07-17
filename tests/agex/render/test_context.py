from agex.llm.core import TextPart
from agex.render.context import ContextRenderer
from agex.state import kv
from agex.state.versioned import Versioned


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
    output_parts = renderer.render(state, budget=100)

    # Combine all text parts into a single string for easy checking
    full_text = "\n".join(
        part.text for part in output_parts if isinstance(part, TextPart)
    )

    assert "x = 1" in full_text
    assert "y = 2" in full_text
    assert "Agent printed:\n" in full_text
    assert "Hello" in full_text
