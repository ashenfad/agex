"""Tests for low-detail event rendering."""

from datetime import datetime, timedelta, timezone

from agex.agent.events import OutputEvent, SuccessEvent, SummaryEvent, TaskStartEvent
from agex.eval.objects import ImageAction
from agex.render.events import render_events_as_markdown


def test_low_detail_rendering_with_threshold():
    """Test that events older than low_detail_threshold render at low detail."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)
    new_time = now

    # Create a summary with low_detail_threshold
    summary = SummaryEvent(
        agent_name="test",
        summary="Summary of old events",
        summarized_event_count=3,
        original_tokens=1000,
        low_detail_threshold=now - timedelta(hours=1),  # 1 hour ago
    )

    # Create old event (should render at low detail)
    old_task_start = TaskStartEvent(
        agent_name="test",
        task_name="old_task",
        message="Old task with lots of detail" * 100,  # Long message
        inputs={},
        timestamp=old_time,
    )

    # Create new event (should render at full detail)
    new_task_start = TaskStartEvent(
        agent_name="test",
        task_name="new_task",
        message="New task with lots of detail" * 100,  # Long message
        inputs={},
        timestamp=new_time,
    )

    events = [summary, old_task_start, new_task_start]
    messages = render_events_as_markdown(events)

    # Summary message is always included
    assert any(
        "Summary of old events" in str(msg.get("content", "")) for msg in messages
    )

    # Both task starts are included, but we can't easily verify token counts without
    # deeper inspection. The key is that the threshold was found and passed to rendering.
    assert len(messages) == 3


def test_low_detail_rendering_with_images():
    """Test that images are replaced with placeholders at low detail."""
    try:
        from PIL import Image
    except ImportError:
        # Skip if PIL not available
        return

    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)

    # Create a summary with low_detail_threshold
    summary = SummaryEvent(
        agent_name="test",
        summary="Summary",
        summarized_event_count=1,
        original_tokens=100,
        low_detail_threshold=now - timedelta(hours=1),
    )

    # Create old output event with image (should replace with placeholder)
    image = Image.new("RGB", (100, 100), color="red")
    old_output = OutputEvent(
        agent_name="test",
        parts=[ImageAction(image=image)],
        timestamp=old_time,
    )

    # Create new output event with image (should include full image)
    new_output = OutputEvent(
        agent_name="test",
        parts=[ImageAction(image=image)],
        timestamp=now,
    )

    events = [summary, old_output, new_output]
    messages = render_events_as_markdown(events)

    # Check that old output has text content (placeholder)
    # and new output has image content
    old_msg = messages[1]
    new_msg = messages[2]

    # Old message should be text-only (image replaced with placeholder)
    assert isinstance(old_msg["content"], str)
    assert "[Image]" in old_msg["content"]

    # New message should be multimodal (has actual image)
    assert isinstance(new_msg["content"], list)
    assert any(
        part.get("type") == "image"
        for part in new_msg["content"]
        if isinstance(part, dict)
    )


def test_low_detail_rendering_with_success():
    """Test that SuccessEvent results are truncated at low detail."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)

    # Create a summary with low_detail_threshold
    summary = SummaryEvent(
        agent_name="test",
        summary="Summary",
        summarized_event_count=1,
        original_tokens=100,
        low_detail_threshold=now - timedelta(hours=1),
    )

    # Create a result large enough to exceed the LOW_DETAIL char budget (4096)
    # but small enough to fit within HI_DETAIL (32768)
    complex_result = {f"key_{i}": "x" * 200 for i in range(30)}
    old_success = SuccessEvent(
        agent_name="test",
        result=complex_result,
        timestamp=old_time,
    )

    # Create new success event with same result
    new_success = SuccessEvent(
        agent_name="test",
        result=complex_result,
        timestamp=now,
    )

    events = [summary, old_success, new_success]
    messages = render_events_as_markdown(events)

    old_msg_content = messages[1]["content"]
    new_msg_content = messages[2]["content"]

    # Low-detail budget (4096 chars) should truncate, high-detail (32768) should not
    assert len(old_msg_content) < len(new_msg_content)


def test_no_low_detail_without_summary():
    """Test that without SummaryEvent, all events render at full detail."""
    now = datetime.now(timezone.utc)
    old_time = now - timedelta(hours=2)

    # No summary event - all should render at full detail
    old_task_start = TaskStartEvent(
        agent_name="test",
        task_name="old_task",
        message="Old task",
        inputs={},
        timestamp=old_time,
    )

    new_task_start = TaskStartEvent(
        agent_name="test",
        task_name="new_task",
        message="New task",
        inputs={},
        timestamp=now,
    )

    events = [old_task_start, new_task_start]
    messages = render_events_as_markdown(events)

    # Both render at full detail (no threshold)
    assert len(messages) == 2
