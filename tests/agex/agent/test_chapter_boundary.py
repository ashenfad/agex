"""Tests for the boundary-based chaptering helpers.

Covers ``build_chapter_scope_filter``, ``build_boundary_index``, and
``has_completable_boundary`` — the new pieces that Phase 3 will wire
into ``_maybe_chapter`` and the renderer's Filter A.
"""

import pytest

from agex import clear_agent_registry
from agex.agent.chapter import (
    CHAPTER_TASK,
    build_boundary_index,
    build_chapter_scope_filter,
    has_completable_boundary,
)
from agex.agent.events import (
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
    FailEvent,
    SuccessEvent,
    TaskStartEvent,
)
from tests.agex._emissions import make_action_event


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ts(name: str = "t1", agent: str = "a") -> TaskStartEvent:
    return TaskStartEvent(
        agent_name=agent,
        task_name=name,
        inputs={"message": f"do {name}"},
        message=f"do {name}",
    )


def _ts_no_msg(name: str, agent: str = "a") -> TaskStartEvent:
    return TaskStartEvent(agent_name=agent, task_name=name, inputs={}, message="")


def _success(result="ok", agent: str = "a") -> SuccessEvent:
    return SuccessEvent(agent_name=agent, result=result)


def _fail(msg="boom", agent: str = "a") -> FailEvent:
    return FailEvent(agent_name=agent, message=msg)


def _chapter(name="ch", message="summary", agent: str = "a") -> ChapterEvent:
    return ChapterEvent(agent_name=agent, name=name, message=message)


# ---------------------------------------------------------------------------
# build_chapter_scope_filter
# ---------------------------------------------------------------------------


class TestBuildChapterScopeFilter:
    def test_empty(self):
        assert build_chapter_scope_filter([]) == set()
        assert build_chapter_scope_filter([], include_open=True) == set()

    def test_no_chapter_task_no_skip(self):
        events = [
            _ts("t1"),
            make_action_event(agent_name="a", thinking="t", code="x"),
            _success(),
        ]
        assert build_chapter_scope_filter(events) == set()
        assert build_chapter_scope_filter(events, include_open=True) == set()

    def test_closed_chapter_scope_marked(self):
        """Renderer mode (default): a closed __chapter__ scope is skipped
        in its entirety — taskStart through the closing success."""
        events = [
            _ts("t1"),  # 0
            _success(),  # 1
            _ts(CHAPTER_TASK),  # 2  open chapter
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
            _success(result=[]),  # 4  close chapter
            _ts("t2"),  # 5
            _success(),  # 6
        ]
        skip = build_chapter_scope_filter(events)
        assert skip == {2, 3, 4}

    def test_open_chapter_scope_visible_to_renderer(self):
        """Renderer mode: an *unclosed* chapter scope is NOT marked, so
        the chapter task's own loop can still see its taskStart prompt."""
        events = [
            _ts("t1"),
            _success(),
            _ts(CHAPTER_TASK),  # opens, no close yet
            make_action_event(agent_name="a", thinking="t", code="x"),
        ]
        skip = build_chapter_scope_filter(events)
        assert skip == set()  # open scope stays visible

    def test_open_chapter_scope_marked_in_index_mode(self):
        """Index-builder mode (include_open=True): even an unclosed
        chapter scope is filtered — the chapter task can't enumerate
        itself as a foldable boundary."""
        events = [
            _ts("t1"),  # 0
            _success(),  # 1
            _ts(CHAPTER_TASK),  # 2
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
        ]
        skip = build_chapter_scope_filter(events, include_open=True)
        assert skip == {2, 3}

    def test_chapter_task_terminator_variants(self):
        """Fail / Cancelled / Clarify all close a chapter scope just
        like Success."""
        for terminator in [
            FailEvent(agent_name="a", message="bad"),
            CancelledEvent(
                agent_name="a", task_name=CHAPTER_TASK, iterations_completed=2
            ),
            ClarifyEvent(agent_name="a", message="?"),
        ]:
            events = [
                _ts(CHAPTER_TASK),
                make_action_event(agent_name="a", thinking="t", code="x"),
                terminator,
                _ts("t2"),
                _success(),
            ]
            skip = build_chapter_scope_filter(events)
            assert skip == {0, 1, 2}, f"failed for {type(terminator).__name__}"

    def test_multiple_closed_chapter_scopes(self):
        events = [
            _ts("t1"),  # 0
            _success(),  # 1
            _ts(CHAPTER_TASK),  # 2
            _success(result=[]),  # 3
            _ts("t2"),  # 4
            _success(),  # 5
            _ts(CHAPTER_TASK),  # 6
            _success(result=[]),  # 7
            _ts("t3"),  # 8
        ]
        skip = build_chapter_scope_filter(events)
        assert skip == {2, 3, 6, 7}

    def test_non_chapter_task_terminator_does_not_close_chapter_scope(self):
        """A regular task's terminator inside a chapter scope shouldn't
        accidentally close the chapter frame.  This exercises the stack
        discipline: regular tasks push 'other' frames so their close
        events pop the right frame."""
        events = [
            _ts(CHAPTER_TASK),  # 0  chapter opens
            _ts("inner"),  # 1  nested regular task
            _success(),  # 2  closes "inner", not chapter
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
            _success(result=[]),  # 4  closes chapter
        ]
        skip = build_chapter_scope_filter(events)
        # Whole chapter scope marked, including the nested task's events.
        assert skip == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# build_boundary_index
# ---------------------------------------------------------------------------


class TestBuildBoundaryIndex:
    def test_empty(self):
        text, ranges = build_boundary_index([])
        assert text == ""
        assert ranges == []

    def test_single_completed_task(self):
        events = [
            _ts("analyze"),
            make_action_event(agent_name="a", thinking="t", code="x"),
            _success(result="found 3 tables"),
        ]
        text, ranges = build_boundary_index(events)
        assert ranges == [(0, 3)]
        lines = text.split("\n")
        assert len(lines) == 1
        assert lines[0].startswith("[1] task ")
        assert "analyze" in lines[0]
        assert "→ success" in lines[0]
        assert "found 3 tables" in lines[0]

    def test_in_progress_task_marked(self):
        events = [
            _ts("running"),
            make_action_event(agent_name="a", thinking="t", code="x"),
        ]
        text, ranges = build_boundary_index(events)
        assert ranges == [(0, 2)]
        assert "(in progress)" in text

    def test_multiple_tasks(self):
        events = [
            _ts("t1"),
            _success(result="r1"),
            _ts("t2"),
            _success(result="r2"),
        ]
        text, ranges = build_boundary_index(events)
        assert ranges == [(0, 2), (2, 4)]
        lines = text.split("\n")
        assert lines[0].startswith("[1] ")
        assert lines[1].startswith("[2] ")
        assert "t1" in lines[0]
        assert "t2" in lines[1]

    def test_chapter_event_is_a_first_class_boundary(self):
        """A prior ChapterEvent appears in the index as its own [N]
        entry — picking a range that includes it folds it into a new
        outer chapter (nested chaptering)."""
        events = [
            _chapter(name="phase 1", message="early findings"),  # 0
            _ts("t2"),  # 1
            _success(result="r2"),  # 2
        ]
        text, ranges = build_boundary_index(events)
        assert ranges == [(0, 1), (1, 3)]
        lines = text.split("\n")
        assert lines[0].startswith("[1] chapter ")
        assert "phase 1" in lines[0]
        assert "early findings" in lines[0]
        assert lines[1].startswith("[2] task ")

    def test_prior_chapter_task_bookkeeping_excluded_from_index(self):
        """The chapter task's own bookkeeping (taskStart + Success) is
        filtered from the boundary *index* — it's not a foldable
        boundary, just framework noise.

        Note the range semantics: each boundary's range extends to the
        next boundary's start, so the filtered chapter-task bookkeeping
        gets absorbed into the *preceding* boundary's range. When the
        agent then folds that boundary, the bookkeeping events are
        swept up into the new chapter (still browsable via the
        ``/chapters/`` VFS overlay). This is intentional — the
        alternative (trimming filtered tails) leaves orphan refs that
        accumulate over repeated chaptering rounds.
        """
        events = [
            _ts("t1"),  # 0
            _success(result="r1"),  # 1
            _ts(CHAPTER_TASK),  # 2  bookkeeping
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
            _success(result=[]),  # 4
            _ts("t2"),  # 5
            _success(result="r2"),  # 6
        ]
        text, ranges = build_boundary_index(events)
        # Two boundaries — t1 and t2.  The chapter-task scope is
        # absorbed into t1's range so its refs get swept up if t1
        # is folded.
        assert ranges == [(0, 5), (5, 7)]
        lines = text.split("\n")
        assert "__chapter__" not in text
        assert lines[0].startswith("[1] task ")
        assert lines[1].startswith("[2] task ")

    def test_chapter_task_in_flight_excluded_from_index(self):
        """In-flight chapter-task bookkeeping is filtered from the
        index even though its scope hasn't closed.

        In normal flow this case doesn't arise (``build_boundary_index``
        is invoked *before* the chapter task starts logging anything).
        Tested here as a defensive property of the helper. The
        in-flight scope gets absorbed into the preceding boundary's
        range — same semantics as the closed case.
        """
        events = [
            _ts("t1"),  # 0
            _success(result="r1"),  # 1
            _ts(CHAPTER_TASK),  # 2  in-progress chapter task
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
        ]
        text, ranges = build_boundary_index(events)
        # Single boundary; in-flight chapter scope swept into t1's range.
        assert ranges == [(0, 4)]
        assert "__chapter__" not in text

    def test_outcome_variants(self):
        events = [
            _ts("ok"),
            _success(result="r"),
            _ts("bad"),
            _fail(msg="boom"),
            _ts("ask"),
            ClarifyEvent(agent_name="a", message="which one?"),
            _ts("kill"),
            CancelledEvent(agent_name="a", task_name="kill", iterations_completed=3),
            _ts("wip"),
            make_action_event(agent_name="a", thinking="t", code="x"),
        ]
        text, _ = build_boundary_index(events)
        lines = text.split("\n")
        assert "→ success" in lines[0]
        assert "→ fail" in lines[1]
        assert "boom" in lines[1]
        assert "→ clarify" in lines[2]
        assert "→ cancelled" in lines[3]
        assert "(in progress)" in lines[4]

    def test_chapter_event_followed_by_task(self):
        """ChapterEvent → TaskStart → Success should give two boundaries
        with the right ranges."""
        events = [
            _chapter(),  # 0
            _ts("t2"),  # 1
            make_action_event(agent_name="a", thinking="t", code="x"),  # 2
            _success(),  # 3
        ]
        _, ranges = build_boundary_index(events)
        assert ranges == [(0, 1), (1, 4)]


# ---------------------------------------------------------------------------
# has_completable_boundary
# ---------------------------------------------------------------------------


class TestHasCompletableBoundary:
    def test_empty(self):
        assert has_completable_boundary([], []) is False

    def test_only_in_progress_task(self):
        events = [
            _ts("running"),
            make_action_event(agent_name="a", thinking="t", code="x"),
        ]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is False

    def test_completed_task_is_completable(self):
        events = [_ts("done"), _success(result="r")]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_failed_task_is_completable(self):
        events = [_ts("bad"), _fail()]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_cancelled_task_is_completable(self):
        events = [
            _ts("kill"),
            CancelledEvent(agent_name="a", task_name="kill", iterations_completed=2),
        ]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_clarified_task_is_completable(self):
        """Per the design discussion: a clarify is a completion (the
        task is closed, awaiting a human), so it counts as foldable."""
        events = [_ts("ask"), ClarifyEvent(agent_name="a", message="?")]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_prior_chapter_alone_is_completable(self):
        """A prior ChapterEvent alongside an in-progress task is enough
        to consider the index foldable — the chapter can be folded
        into a deeper chapter even without other completed tasks."""
        events = [
            _chapter(),  # 0  prior chapter
            _ts("running"),  # 1  in-progress
            make_action_event(agent_name="a", thinking="t", code="x"),  # 2
        ]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_running_task_with_completed_predecessor_is_completable(self):
        events = [
            _ts("done"),
            _success(),
            _ts("running"),
            make_action_event(agent_name="a", thinking="t", code="x"),
        ]
        _, ranges = build_boundary_index(events)
        assert has_completable_boundary(events, ranges) is True

    def test_running_parent_with_closed_chapter_scope_inside_is_not_completable(self):
        """A parent task is in-progress; a chapter task ran inside its
        range and closed.  The parent's range absorbs the chapter
        scope, so the chapter task's Success event sits inside it.
        ``has_completable_boundary`` must skip events inside chapter
        scopes — otherwise it'd falsely report the running parent as
        completable.
        """
        events = [
            _ts("running"),  # 0  parent in-progress
            make_action_event(agent_name="a", thinking="t", code="x"),  # 1
            _ts(CHAPTER_TASK),  # 2  chapter scope opens
            make_action_event(agent_name="a", thinking="t", code="x"),  # 3
            _success(result=[]),  # 4  closes chapter scope (NOT parent)
            make_action_event(
                agent_name="a", thinking="t", code="x"
            ),  # 5  parent still going
        ]
        _, ranges = build_boundary_index(events)
        assert ranges == [(0, 6)]
        # No real terminator for the parent — chapter task's Success
        # is filtered out of the scan.
        assert has_completable_boundary(events, ranges) is False

    def test_only_in_flight_chapter_task_plus_running_parent(self):
        """When the only "completed" entries in the raw log are the
        chapter task's own (filtered) bookkeeping plus an in-progress
        parent, has_completable_boundary should return False — there's
        nothing the chapter task can legitimately fold."""
        events = [
            _ts("running"),  # 0  in-progress parent
            _ts(CHAPTER_TASK),  # 1  in-flight chapter task
            make_action_event(agent_name="a", thinking="t", code="x"),  # 2
        ]
        _, ranges = build_boundary_index(events)
        # Single boundary (the running parent); the in-flight chapter
        # scope is absorbed into its range (no terminator yet).
        assert ranges == [(0, 3)]
        # No terminator anywhere — nothing foldable.
        assert has_completable_boundary(events, ranges) is False
