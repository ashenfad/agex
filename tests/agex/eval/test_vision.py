"""
Tests for vision capabilities, including the `view_image` built-in.
"""

from typing import Any

from agex.agent import Agent, clear_agent_registry
from agex.llm.core import ImagePart, MultimodalMessage
from agex.llm.dummy_client import DummyLLMClient, LLMResponse
from agex.state.kv import Memory
from agex.state.versioned import Versioned

# Try to import Pillow for creating a test image
try:
    from PIL import Image
except ImportError:
    Image = None


def test_view_image_sends_multimodal_message():
    """
    Tests that calling `view_image` results in a MultimodalMessage
    with an ImagePart being sent to the LLM.
    """
    if Image is None:
        # Skip this test if Pillow is not installed
        return

    clear_agent_registry()
    agent = Agent(name="test_agent", max_iterations=3)

    # We need to capture the messages sent to the LLM client and provide a response.
    # The first response from the LLM will call view_image.
    # The second response will see the rendered image and finish the task.
    llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will view the image provided in the inputs.",
                code="view_image(inputs.img_to_view)",
            ),
            LLMResponse(
                thinking="I have now seen the image and will finish.",
                code="task_success('done')",
            ),
        ]
    )
    agent.llm_client = llm_client

    # Create a simple black 10x10 image for the test
    test_image = Image.new("RGB", (10, 10), "black")

    @agent.task
    def view_image_task(img_to_view: Any) -> str:  # type: ignore[return-value]
        """A task that calls view_image."""
        pass

    state = Versioned(Memory())
    result = view_image_task(test_image, state=state)

    assert result == "done"

    # Check the messages that were actually sent to the LLM
    sent_messages = llm_client.all_messages
    assert len(sent_messages) > 0

    # The message containing the image is the user message in the *second* call to the LLM.
    second_llm_call_messages = sent_messages[1]
    last_user_message = second_llm_call_messages[-1]

    # It should be a MultimodalMessage
    assert isinstance(last_user_message, MultimodalMessage)

    # Its content should contain an ImagePart
    content = last_user_message.content
    has_image_part = any(isinstance(part, ImagePart) for part in content)
    assert has_image_part, "The message should contain an ImagePart."
