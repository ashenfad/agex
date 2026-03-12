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

IMPORTANT: Not everything should be chaptered. Leave recent and ongoing work in \
full context — you need those details to continue effectively. Only chapter work \
that is clearly finished and whose full details you no longer need at hand. When \
in doubt, leave it unchaptered. It is perfectly fine to return an empty list.

Rules:
- start and end are 1-based inclusive indices from the event index
- Ranges must be contiguous and non-overlapping
- The message should distill key details from the full event content, not just \
restate the index summaries
- Return task_success([chapter1, chapter2, ...]) with your chapters
- Return task_success([]) if nothing can be chaptered right now

Good chapters:
- Capture the key findings, decisions, data, and outcomes from the full context
- Use descriptive names that serve as a mini table of contents
- Close out completed phases — never chapter work in progress or the most recent task
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


def prepare_events_for_chaptering(
    all_events: list[BaseEvent],
) -> tuple[list[BaseEvent], list[int]]:
    """Filter events for the chaptering index.

    Excludes ``ErrorEvent`` instances and any events belonging to prior
    ``__chapter__`` tasks (TaskStart through terminal event) so the
    chaptering agent only sees substantive work.

    Returns:
        visible_events: Events suitable for :func:`build_numbered_event_index`.
        visible_to_log: Mapping from visible index to ``all_events`` index,
            used to convert the agent's 1-based chapter ranges back to log
            positions.
    """
    from agex.agent.events import ErrorEvent

    visible_events: list[BaseEvent] = []
    visible_to_log: list[int] = []
    in_chapter_task = False
    for log_idx, event in enumerate(all_events):
        if isinstance(event, TaskStartEvent):
            in_chapter_task = event.task_name == CHAPTER_TASK
        if not isinstance(event, ErrorEvent) and not in_chapter_task:
            visible_events.append(event)
            visible_to_log.append(log_idx)
        if in_chapter_task and isinstance(
            event, (SuccessEvent, FailEvent, CancelledEvent, ClarifyEvent)
        ):
            in_chapter_task = False
    return visible_events, visible_to_log


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
    events: list[BaseEvent], chaptering_trigger: int | None
) -> bool:
    """Check whether chaptering should be triggered.

    Uses the most recent ActionEvent's input_tokens (actual API token count)
    to determine if the context has exceeded the chaptering trigger.

    Args:
        events: List of events from the log.
        chaptering_trigger: Token threshold to trigger chaptering.
            If None, chaptering is disabled.

    Returns:
        True if chaptering should be triggered.
    """
    if chaptering_trigger is None:
        return False
    latest = get_latest_input_tokens(events)
    return latest is not None and latest > chaptering_trigger
