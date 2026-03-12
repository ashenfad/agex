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
    ClarifyEvent,
    FailEvent,
    SuccessEvent,
    TaskStartEvent,
)

CHAPTER_TASK = "__chapter__"


@dataclass
class Chapter:
    """A chapter to close over a contiguous range of tasks.

    The agent creates Chapter instances and returns them via task_success()
    to request chaptering. The framework converts them to ChapterEvents.

    Attributes:
        start: 1-based inclusive start task number from the task index.
        end: 1-based inclusive end task number from the task index.
        name: Short descriptive name for the chapter.
        message: Agent's summary/distillation of the chaptered work.
    """

    start: int
    end: int
    name: str
    message: str


CHAPTER_TASK_PRIMER = """\
You are being asked to manage your context by closing completed tasks \
into named chapters. Each chapter replaces a contiguous range of tasks with a \
concise summary, while preserving the originals for later browsing.

You will receive a task index mapping numbered tasks to your full context above. \
The numbers in the index correspond to the [N] prefixes on the task starts in your \
conversation. Use the index to identify ranges, but read the full task content \
in your context to write thorough summaries that capture important details.

Create Chapter instances to close out completed tasks:

    Chapter(start=1, end=3, name="Data exploration", message="Found 3 tables...")

IMPORTANT: Not everything should be chaptered. Leave recent and ongoing work in \
full context — you need those details to continue effectively. Only chapter tasks \
that are clearly finished and whose full details you no longer need at hand. When \
in doubt, leave them unchaptered. It is perfectly fine to return an empty list.

Rules:
- start and end are 1-based inclusive task numbers from the task index
- Ranges must be contiguous and non-overlapping
- The message should distill key details from the full task content, not just \
restate the index summaries
- Return task_success([chapter1, chapter2, ...]) with your chapters
- Return task_success([]) if nothing can be chaptered right now

Good chapters:
- Capture the key findings, decisions, data, and outcomes from the full context
- Use descriptive names that serve as a mini table of contents
- Close out completed phases — never chapter work in progress or the most recent task
"""


@dataclass
class _TaskInfo:
    """Internal: metadata about a single task for the chaptering index."""

    name: str
    message: str
    outcome: str | None
    complete: bool
    log_start: int
    log_end: int


def prepare_tasks_for_chaptering(
    all_events: list[BaseEvent],
) -> tuple[list[_TaskInfo], list[tuple[int, int]]]:
    """Group events into tasks for the chaptering index.

    Walks the event log and groups events by ``TaskStartEvent`` boundaries.
    Each task group spans from its ``TaskStartEvent`` to just before the next
    ``TaskStartEvent`` (or the end of the log).

    Returns:
        tasks: List of :class:`_TaskInfo` with task metadata.
        task_to_log_range: Parallel list of ``(log_start, log_end)`` tuples
            (0-based, end-exclusive) covering each task's events in the log.
    """
    from agex.agent.events import ErrorEvent

    tasks: list[_TaskInfo] = []
    current: _TaskInfo | None = None

    for log_idx, event in enumerate(all_events):
        if isinstance(event, ErrorEvent):
            continue

        if isinstance(event, TaskStartEvent):
            # Close previous task
            if current is not None:
                current.log_end = log_idx
                tasks.append(current)

            msg = event.inputs.get("message", str(event.inputs)) if event.inputs else ""
            current = _TaskInfo(
                name=event.task_name,
                message=msg,
                outcome=None,
                complete=False,
                log_start=log_idx,
                log_end=log_idx,
            )

        elif isinstance(event, SuccessEvent) and current is not None:
            result_str = str(event.result)[:100]
            if len(str(event.result)) > 100:
                result_str += "..."
            current.outcome = result_str
            current.complete = True

        elif isinstance(event, FailEvent) and current is not None:
            current.outcome = f"Failed: {event.message[:80]}"
            current.complete = True

        elif isinstance(event, CancelledEvent) and current is not None:
            current.outcome = "Cancelled"
            current.complete = True

        elif isinstance(event, ClarifyEvent) and current is not None:
            current.outcome = f"Clarify: {event.message[:80]}"
            current.complete = True

    # Close last task
    if current is not None:
        current.log_end = len(all_events)
        tasks.append(current)

    task_to_log_range = [(t.log_start, t.log_end) for t in tasks]
    return tasks, task_to_log_range


def build_numbered_task_index(tasks: list[_TaskInfo]) -> str:
    """Build a compact numbered index of tasks for the chapter task.

    Each line shows the task name/input and its outcome. The numbers
    correspond to the ``[N]`` prefixes on task starts in the rendered context.

    Args:
        tasks: List of :class:`_TaskInfo` from :func:`prepare_tasks_for_chaptering`.

    Returns:
        Multi-line string with numbered task summaries.
    """
    lines = []
    for i, task in enumerate(tasks, 1):
        label = f'"{task.name}"'
        if task.message:
            msg = task.message[:60].replace("\n", " ")
            if len(task.message) > 60:
                msg += "..."
            label += f": {msg}"

        if not task.complete:
            lines.append(f"[{i}] {label} (in progress)")
        elif task.outcome:
            outcome = task.outcome.replace("\n", " ")
            lines.append(f"[{i}] {label} → {outcome}")
        else:
            lines.append(f"[{i}] {label}")
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
