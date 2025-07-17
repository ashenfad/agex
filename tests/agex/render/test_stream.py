from agex.llm.core import TextPart
from agex.render.stream import StreamRenderer


def test_state_budget_truncation():
    """Checks that the state stream is truncated independently."""
    renderer = StreamRenderer("gpt-4o")

    # Budget is not enough for the long string, but enough for the short one and marker.
    output = renderer.render_state_stream(
        items={
            "x": "a" * 100,  # This will be truncated
            "y": 2,  # This should be visible
        },
        budget=20,
    )
    assert "y = 2" in output
    assert "..." in output
    assert "a" * 50 not in output


def test_large_collection_in_state():
    """Checks that large collections in state are summarized."""
    renderer = StreamRenderer("gpt-4o")
    output = renderer.render_state_stream(items={"x": list(range(10000))}, budget=100)
    assert "x = [... (10000 items)]" in output


def test_stdout_budget_truncation():
    """Checks that the stdout stream is truncated independently."""
    renderer = StreamRenderer("gpt-4o")
    output_parts = renderer.render_item_stream(
        items=[
            "This is a very long line that will definitely be truncated because it is extremely long",
            "This should be visible",
        ],
        budget=20,
    )
    full_text = "\n".join(
        part.text for part in output_parts if isinstance(part, TextPart)
    )

    assert "This should be visible" in full_text
    assert "..." in full_text
    assert "very long line" not in full_text


def test_large_collection_in_stdout():
    """Checks that large collections in stdout are summarized."""
    renderer = StreamRenderer("gpt-4o")
    output_parts = renderer.render_item_stream(
        items=[list(range(10000)), "visible"], budget=100
    )
    full_text = "\n".join(
        part.text for part in output_parts if isinstance(part, TextPart)
    )

    assert "[... (10000 items)]" in full_text
    assert "visible" in full_text
