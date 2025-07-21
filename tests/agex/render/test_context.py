from agex.agent.events import OutputEvent
from agex.llm.core import TextPart
from agex.render.context import ContextRenderer
from agex.state import kv
from agex.state.versioned import Versioned


def test_render_events():
    """
    Checks that ContextRenderer correctly renders a list of events.
    """
    event1 = OutputEvent(agent_name="test", parts=["Hello"])
    event2 = OutputEvent(agent_name="test", parts=[{"data": 123}])
    events = [event1, event2]

    renderer = ContextRenderer("gpt-4o")
    # High budget, everything should be visible.
    output_parts = renderer.render_events(events, budget=100)

    # Combine all text parts into a single string for easy checking
    full_text = "\n".join(
        part.text for part in output_parts if isinstance(part, TextPart)
    )

    assert "Agent printed:" in full_text
    assert "Hello" in full_text
    assert "{'data': 123}" in full_text


def test_render_state_changes():
    """
    Checks that ContextRenderer correctly renders state changes.
    """
    state = Versioned(kv.Memory())

    # Make some changes and snapshot
    state.set("x", 42)
    state.set("message", "Hello World")
    state.snapshot()

    # Make more changes
    state.set("y", [1, 2, 3])
    state.set("x", 100)  # Update existing value
    state.snapshot()  # Need to snapshot to see the changes in diffs()

    renderer = ContextRenderer("gpt-4o")
    output_parts = renderer.render(state, budget=200)

    # Should render recent state changes
    if output_parts:
        full_text = "\n".join(
            part.text for part in output_parts if isinstance(part, TextPart)
        )

        assert "x = 100" in full_text  # Updated value
        assert "y = [1, 2, 3]" in full_text  # New value
        # Should not show the old x = 42 since it was overwritten
