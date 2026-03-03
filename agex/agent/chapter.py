"""
Chapter support for agent-directed context compaction.

Provides the Chapter dataclass, the chapter task primer, and helper utilities
for building numbered event indices and triggering chaptering.
"""

from dataclasses import dataclass

from agex.agent.events import (
    ActionEvent,
    BaseEvent,
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
    FailEvent,
    FileEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.eval.objects import PrintAction

CHAPTER_TASK = "__chapter__"


@dataclass
class Chapter:
    """A chapter to close over a contiguous range of events.

    The agent creates Chapter instances and returns them via task_success()
    to request chaptering. The framework converts them to ChapterEvents.

    Attributes:
        start: 1-based inclusive start index into the event log.
        end: 1-based inclusive end index into the event log.
        name: Short descriptive name for the chapter.
        message: Agent's summary/distillation of the chaptered events.
    """

    start: int
    end: int
    name: str
    message: str


CHAPTER_TASK_PRIMER = """\
You are being asked to manage your context by closing completed stretches of work \
into named chapters. Each chapter replaces a contiguous range of events with a \
concise summary, while preserving the originals for later browsing.

You will receive an event index mapping numbered events to your full context above. \
The numbers in the index correspond to the [N] prefixes on each event in your \
conversation. Use the index to identify ranges, but read the full event content \
in your context to write thorough summaries that capture important details.

Create Chapter instances to close out completed work:

    Chapter(start=1, end=4, name="Data exploration", message="Found 3 tables...")

Rules:
- start and end are 1-based inclusive indices from the event index
- Ranges must be contiguous and non-overlapping
- The message should distill key details from the full event content, not just \
restate the index summaries
- Chapter completed stretches of work; keep active/recent work unchaptered
- Return task_success([chapter1, chapter2, ...]) with your chapters
- Return task_success([]) if nothing can be chaptered right now

Good chapters:
- Capture the key findings, decisions, data, and outcomes from the full context
- Use descriptive names that serve as a mini table of contents
- Close out phases of work that are done, not work in progress
"""


def _summarize_event(event: BaseEvent) -> str:
    """Produce a compact one-line summary of an event for the numbered index."""
    if isinstance(event, TaskStartEvent):
        return f'Task: "{event.task_name}"'

    if isinstance(event, ActionEvent):
        title = event.title or "untitled"
        # Show whether it was code or terminal
        if event.terminal:
            return f"Action: {title} (terminal)"
        lines = len((event.code or "").strip().splitlines())
        return f"Action: {title} ({lines} lines)"

    if isinstance(event, OutputEvent):
        n = len(event.parts)
        if n == 0:
            return "Output: (empty)"
        # Show first text snippet
        for part in event.parts:
            if isinstance(part, PrintAction) and part:
                first_arg = str(part[0])
                snippet = first_arg[:60]
                if len(first_arg) > 60:
                    snippet += "..."
                return f"Output: {snippet}"
        return f"Output: ({n} parts)"

    if isinstance(event, SuccessEvent):
        result_str = str(event.result)[:60]
        if len(str(event.result)) > 60:
            result_str += "..."
        return f"Success: {result_str}"

    if isinstance(event, FailEvent):
        return f"Fail: {event.message[:60]}"

    if isinstance(event, ClarifyEvent):
        return f"Clarify: {event.message[:60]}"

    if isinstance(event, CancelledEvent):
        return f"Cancelled: {event.task_name}"

    if isinstance(event, ChapterEvent):
        return f'Chapter: "{event.name}" — {event.message[:50]}'

    if isinstance(event, FileEvent):
        parts = []
        if event.added:
            parts.append(f"+{len(event.added)}")
        if event.modified:
            parts.append(f"~{len(event.modified)}")
        if event.removed:
            parts.append(f"-{len(event.removed)}")
        return f"Files: {' '.join(parts)}"

    return f"{type(event).__name__}"


def build_numbered_event_index(events: list[BaseEvent]) -> str:
    """Build a compact numbered index of events for the chapter task.

    Each line is a one-line summary with a 1-based index. The input should
    be pre-filtered (e.g. ErrorEvents removed) so that indices match the
    [N] prefixes agents see in rendered context.

    Args:
        events: Pre-filtered list of visible events.

    Returns:
        Multi-line string with numbered event summaries.
    """
    lines = []
    for i, event in enumerate(events, 1):
        summary = _summarize_event(event).replace("\n", " ")
        lines.append(f"[{i}] {summary}")
    return "\n".join(lines)


def get_latest_input_tokens(events: list[BaseEvent]) -> int | None:
    """Return input_tokens from the most recent ActionEvent, or None."""
    for event in reversed(events):
        if isinstance(event, ActionEvent) and event.input_tokens is not None:
            return event.input_tokens
    return None


def should_trigger_chaptering(
    events: list[BaseEvent], high_water_tokens: int | None
) -> bool:
    """Check whether chaptering should be triggered.

    Uses the most recent ActionEvent's input_tokens (actual API token count)
    to determine if the context has grown past the high water mark.

    Args:
        events: List of events from the log.
        high_water_tokens: Token threshold to trigger chaptering.
            If None, chaptering is disabled.

    Returns:
        True if chaptering should be triggered.
    """
    if high_water_tokens is None:
        return False
    latest = get_latest_input_tokens(events)
    return latest is not None and latest > high_water_tokens
