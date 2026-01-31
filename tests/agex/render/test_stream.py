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


def test_output_parts_exceeding_budget_still_returns_content():
    """
    Regression test: ensure that a single large output that exceeds the token
    budget is truncated to fit rather than being silently dropped.

    Previously, if a PrintAction's content exceeded the token budget after
    ValueRenderer truncation, the entire output was dropped and the agent
    would see nothing - causing it to potentially retry indefinitely.
    """
    from agex.eval.objects import PrintAction
    from agex.render.primitives import HI_DETAIL_BUDGET, render_output_parts_full

    # Create content that mimics minified HTML - lots of small tokens
    # HTML has ~3 chars/token vs simple text at ~4 chars/token
    # This causes char-truncated output to still exceed token budget
    html_chunk = '<div class="item" id="c123" style="color: red;">'
    huge_output = html_chunk * 3000  # ~140KB of HTML-like content

    parts = [PrintAction([huge_output])]
    content_parts, token_count = render_output_parts_full(
        parts, budget=HI_DETAIL_BUDGET
    )

    # Must return something, not empty!
    assert len(content_parts) > 0, "Output should not be silently dropped"
    assert token_count > 0, "Token count should be non-zero"
    assert token_count <= HI_DETAIL_BUDGET, "Should respect budget"

    # The output should contain some of the original content
    text = content_parts[0].text
    assert "div" in text, "Should contain some original content"


def test_output_parts_with_tiny_budget_shows_placeholder():
    """
    If the budget is so small that even truncation doesn't help,
    we should still show a placeholder message rather than nothing.
    """
    from agex.eval.objects import PrintAction
    from agex.render.primitives import render_output_parts_full

    huge_output = "x" * 10000
    parts = [PrintAction([huge_output])]

    # Very tiny budget - can't fit meaningful content
    content_parts, token_count = render_output_parts_full(parts, budget=50)

    # Should still return something
    assert len(content_parts) > 0, "Should show placeholder even with tiny budget"
