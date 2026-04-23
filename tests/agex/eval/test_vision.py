"""
Tests for vision capabilities, including the `view_image` built-in.
"""

from typing import Any

from agex.agent import Agent, clear_agent_registry
from agex.agent.events import OutputEvent
from agex.llm.dummy_client import Dummy
from agex.state import connect_state
from tests.agex._emissions import make_response

# Try to import Pillow for creating a test image
try:
    from PIL import Image
except ImportError:
    Image = None


def test_view_image_sends_image_in_output_event():
    """
    Tests that calling `view_image` works end-to-end with the event-based interface.
    """
    if Image is None:
        # Skip this test if Pillow is not installed
        return

    clear_agent_registry()
    # We need to capture the events sent to the LLM client and provide a response.
    # The first response from the LLM will call view_image.
    # The second response will see the rendered image and finish the task.
    llm = Dummy(
        responses=[
            make_response(
                thinking="I will view the image provided in the inputs.",
                code="view_image(inputs.img_to_view);task_continue();",
            ),
            make_response(
                thinking="I have now seen the image and will finish.",
                code="task_success('done')",
            ),
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="test_agent", max_iterations=3, llm=llm, state=config)

    # Create a simple black 10x10 image for the test
    test_image = Image.new("RGB", (10, 10), "black")

    @agent.task
    def view_image_task(img_to_view: Any) -> str:  # type: ignore[return-value]
        """A task that calls view_image."""
        pass

    result = view_image_task(test_image, session="test_session")

    # The main test: the task should complete successfully
    assert result == "done"

    # Verify that events were sent to the LLM
    sent_events = llm.all_events
    assert len(sent_events) >= 2, "Should have at least 2 LLM calls"

    # Verify that the second call contains OutputEvents (which would include the image)
    second_llm_call_events = sent_events[1]
    output_events = [e for e in second_llm_call_events if isinstance(e, OutputEvent)]
    assert len(output_events) > 0, "Second LLM call should contain OutputEvents"
