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
Compact your context by folding completed work into named chapters. \
You were invoked because your context is over budget — default to folding \
something. The originals stay browsable at /chapters/<slug>/.

The numbered index in your inputs maps to the [N] boundaries you can fold. \
Each entry is either a task you ran (with its outcome) or a chapter you \
produced earlier. Read the full task content in your context above to write \
detailed summaries; the index is just for referring to ranges.

Construct Chapter instances and return them via task_success:

    task_success([
        Chapter(start=1, end=3, name="Data exploration",
                message="Found 3 tables: users, orders, products. ..."),
    ])

Fold completed work that's no longer your immediate context. Including a \
prior chapter entry in a new range is normal — that's how you fold older \
summaries into higher-level ones (nested chaptering). Don't fold the \
in-progress entry, or anything you still need detailed access to for active \
work. task_success([]) is a last resort — return it only when every boundary \
is in-progress or actively needed.

Rules:
- start and end are 1-based inclusive boundary positions from the index above.
- Ranges must be contiguous and non-overlapping.
- message must be VERBOSE — capture specific findings, data values, variable \
names, file paths, decisions, and outcomes. The chapter message is what \
you'll see in place of the originals, so include everything you might need \
later.
- name should serve as a table-of-contents entry.
"""


def get_latest_input_tokens(events: list[BaseEvent]) -> int | None:
    """Return input_tokens from the most recent ActionEvent, or None."""
    for event in reversed(events):
        if isinstance(event, ActionEvent) and event.input_tokens is not None:
            return event.input_tokens
    return None


# ============================================================================
# Boundary-based index — used by ``_maybe_chapter`` to enumerate foldable
# boundaries for the chapter task, and by the renderer's Filter A.
# ============================================================================


def build_chapter_scope_filter(
    events: list[BaseEvent], include_open: bool = False
) -> set[int]:
    """Mark log indices that fall inside a ``__chapter__`` task's scope.

    A "chapter scope" is the contiguous run of events from a chapter task's
    ``TaskStartEvent`` through its closing terminator (``Success`` /
    ``Fail`` / ``Cancelled`` / ``Clarify``) — i.e. the bookkeeping the
    chapter task itself produces while running.

    Two callers, two contracts (controlled by ``include_open``):

    * **Renderer (Filter A, ``include_open=False``):** mark only *closed*
      chapter scopes. The currently-running chapter task's events stay
      unmarked so its own loop's ``render_events`` call can see its
      taskStart prompt and any prior turns. Once the chapter task closes,
      the parent's next render skips the now-closed scope.
    * **Index builder (Filter B, ``include_open=True``):** mark events
      inside *both* open and closed chapter scopes. The boundary index
      handed to a chapter task should never enumerate the chapter task's
      own (in-progress) bookkeeping as a foldable boundary — the chapter
      task can't chapter itself.
    """
    skip: set[int] = set()
    # Stack frames are tuples: ("chapter", start_idx) or ("other",).
    # Non-chapter frames are tracked so terminator events pop the right
    # frame in nested cases (e.g. a task inside a chapter scope).
    stack: list[tuple] = []

    def in_chapter_range() -> bool:
        return any(f[0] == "chapter" for f in stack)

    for i, event in enumerate(events):
        # Update the stack based on this event first so that open-scope
        # marking below sees the current event's effect on the stack.
        if isinstance(event, TaskStartEvent):
            if event.task_name == CHAPTER_TASK:
                stack.append(("chapter", i))
            else:
                stack.append(("other",))
        elif isinstance(event, (SuccessEvent, FailEvent, CancelledEvent, ClarifyEvent)):
            if stack:
                top = stack.pop()
                if top[0] == "chapter":
                    # Closed scope — mark from start through this close
                    # event (inclusive).
                    for j in range(top[1], i + 1):
                        skip.add(j)

        # Open-scope marking — only when the caller wants in-progress
        # chapter scopes filtered too (Filter B).  For Filter A this
        # stays off so the running chapter task can see its own loop
        # history.  After a close event the pop has already happened
        # above, so ``in_chapter_range()`` is correctly false here and
        # we don't double-mark the close index.
        if include_open and in_chapter_range():
            skip.add(i)

    return skip


@dataclass
class _BoundaryRange:
    """Internal: the (start, end) log slice owned by a single boundary."""

    start: int  # 0-based, inclusive — the boundary event itself
    end: int  # 0-based, exclusive — next boundary's start, or len(events)


def build_boundary_index(
    events: list[BaseEvent],
) -> tuple[str, list[tuple[int, int]]]:
    """Build the numbered index handed to the chapter task, plus the
    parallel list of underlying log ranges that boundary positions
    resolve to.

    Boundaries are every :class:`TaskStartEvent` (excluding those inside
    a ``__chapter__`` scope, via Filter B) and every :class:`ChapterEvent`.
    Each boundary owns the events from itself up to but not including
    the next boundary; the final boundary owns through the end of the
    log. Picking a range that spans a prior :class:`ChapterEvent` folds
    it into a new outer chapter (nested chaptering).

    Args:
        events: Events from the log.

    Returns:
        ``(text, ranges)`` where:
        * ``text`` is the multi-line numbered index for the chapter
          task's LLM input.
        * ``ranges`` is a parallel list of ``(log_start, log_end)``
          tuples — 0-based, end-exclusive — covering each boundary's
          underlying events.
    """
    skip = build_chapter_scope_filter(events, include_open=True)

    # Locate boundary indices in order.
    boundary_indices: list[int] = []
    for i, event in enumerate(events):
        if i in skip:
            continue
        if isinstance(event, (TaskStartEvent, ChapterEvent)):
            boundary_indices.append(i)

    # Compute (start, end) for each boundary.
    ranges: list[tuple[int, int]] = []
    for i, start in enumerate(boundary_indices):
        end = boundary_indices[i + 1] if i + 1 < len(boundary_indices) else len(events)
        ranges.append((start, end))

    # Render index lines.
    lines: list[str] = []
    for i, (start, end) in enumerate(ranges, 1):
        boundary = events[start]
        label = _describe_boundary(boundary, events, start, end, skip)
        lines.append(f"[{i}] {label}")

    return "\n".join(lines), ranges


def has_completable_boundary(
    events: list[BaseEvent], ranges: list[tuple[int, int]]
) -> bool:
    """True if at least one boundary in ``ranges`` represents foldable
    content — a :class:`ChapterEvent` (always foldable) or a
    :class:`TaskStartEvent` whose range contains a closing terminator
    (``Success`` / ``Fail`` / ``Cancelled`` / ``Clarify``). The
    currently-running task is *not* completable — its boundary range
    has no closing event yet.

    Used by ``_maybe_chapter`` as a runtime guard: when no boundary is
    completable the chapter task is not invoked at all, avoiding a
    wasted LLM call (it would return ``[]``) and the bookkeeping
    pollution that follows.

    Note: chapter-scope events are skipped during the terminator scan.
    Otherwise a parent task's range (which absorbs trailing filtered
    events from a prior chapter task) would falsely look completable
    because the *chapter task's* Success / Fail event sits inside it.
    """
    skip = build_chapter_scope_filter(events, include_open=True)
    for start, end in ranges:
        head = events[start]
        if isinstance(head, ChapterEvent):
            return True
        for j in range(start + 1, end):
            if j in skip:
                continue
            ev = events[j]
            if isinstance(ev, (SuccessEvent, FailEvent, CancelledEvent, ClarifyEvent)):
                return True
    return False


def _describe_boundary(
    boundary: BaseEvent,
    events: list[BaseEvent],
    start: int,
    end: int,
    skip: set[int],
) -> str:
    """Render a single boundary as a one-line index entry."""
    if isinstance(boundary, ChapterEvent):
        name = _truncate(boundary.name, 60)
        msg = _truncate(boundary.message.replace("\n", " "), 80)
        return f'chapter "{name}" — {msg}'

    if not isinstance(boundary, TaskStartEvent):
        return "unknown"

    head = f'task "{_truncate(boundary.task_name, 50)}"'
    # Match the existing label style: pull a short summary from the
    # task's named-input dict if present, else the full inputs repr.
    raw_msg = (
        boundary.inputs.get("message", str(boundary.inputs)) if boundary.inputs else ""
    )
    trailer = ""
    if raw_msg:
        trailer = f": {_truncate(raw_msg.replace(chr(10), ' '), 80)}"

    # Scan the boundary's range for the first closing terminator.
    for j in range(start + 1, end):
        if j in skip:
            continue
        ev = events[j]
        if isinstance(ev, SuccessEvent):
            outcome = _truncate(str(ev.result).replace("\n", " "), 80)
            return f"{head}{trailer} → success: {outcome}"
        if isinstance(ev, FailEvent):
            return f'{head}{trailer} → fail "{_truncate(ev.message, 60)}"'
        if isinstance(ev, ClarifyEvent):
            return f'{head}{trailer} → clarify "{_truncate(ev.message, 60)}"'
        if isinstance(ev, CancelledEvent):
            return f"{head}{trailer} → cancelled"
    return f"{head}{trailer} (in progress)"


def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 1] + "…"


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
