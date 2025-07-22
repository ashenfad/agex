from agex.agent.events import OutputEvent
from agex.llm.core import TextPart
from agex.render.context import ContextRenderer


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
