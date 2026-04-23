"""End-to-end tests for the <REPORT> feature.

Covers:
- ActionEvent.report round-trips through the loop.
- Sub-agent REPORTs are injected into the parent's log as synthetic
  OutputEvents (single child, multiple children, nested, failure).
- Top-level invocations emit no stdout side effects.
- Agents see their own prior REPORTs in rendered history.
"""

import io
from contextlib import redirect_stderr, redirect_stdout

from agex import Agent, clear_agent_registry
from agex.agent.events import ActionEvent, OutputEvent
from agex.eval.objects import PrintAction
from agex.llm.dummy_client import Dummy
from agex.state import connect_state
from agex.state.log import get_events_from_log
from tests.agex._emissions import (
    event_report,
    make_action_event,
    make_response,
)


def _output_event_texts(events):
    """Flatten all PrintAction string content from OutputEvents."""
    out = []
    for ev in events:
        if isinstance(ev, OutputEvent):
            for part in ev.parts:
                if isinstance(part, PrintAction):
                    out.extend(str(x) for x in part)
    return out


class TestReportOnActionEvent:
    """Basic: LLMResponse.report round-trips to ActionEvent.report."""

    def test_action_event_carries_report(self):
        clear_agent_registry()
        state = connect_state(type="versioned", storage="memory")
        agent = Agent(name="reporter", state=state)

        @agent.task("Say something")
        def do_it() -> str:  # type: ignore
            """Task with a REPORT."""
            pass

        agent.llm = Dummy(
            responses=[
                make_response(
                    thinking="I'll tell the user what I'm doing.",
                    report="Working on it now",
                    code="task_success('done')",
                )
            ]
        )

        collected: list = []
        result = do_it(on_event=collected.append)
        assert result == "done"

        actions = [e for e in collected if isinstance(e, ActionEvent)]
        assert len(actions) >= 1
        assert any(event_report(a) == "Working on it now" for a in actions)

    def test_action_event_report_rendered_in_history(self):
        """Agent's own prior REPORT surfaces as an assistant text block
        in rendered history so later turns can read their own prior
        status updates."""
        from agex.llm.formats.tool_use.renderer import render_events_as_tool_use

        clear_agent_registry()
        connect_state(type="versioned", storage="memory")

        action_with_report = make_action_event(
            agent_name="history_agent",
            thinking="working",
            report="Working on it now",
            code="print('x')",
        )
        messages = render_events_as_tool_use([action_with_report])
        assistant = next(m for m in messages if m["role"] == "assistant")
        # The report becomes a ``text`` block in the assistant's
        # content, alongside the python_action tool_use block.
        text_blocks = [b for b in assistant["content"] if b.get("type") == "text"]
        tool_uses = [b for b in assistant["content"] if b.get("type") == "tool_use"]
        assert any(b["text"] == "Working on it now" for b in text_blocks)
        assert tool_uses and tool_uses[0]["name"] == "python_action"


class TestSubAgentReportPropagation:
    """Sub-agent REPORTs are injected into the parent's state log."""

    def _setup_parent_child(self):
        clear_agent_registry()
        state_cfg = connect_state(type="versioned", storage="memory")
        parent = Agent(name="parent", state=state_cfg)
        child = Agent(name="child", state=state_cfg)

        @parent.fn(docstring="Have the child do work and return a value")
        @child.task("Do the work")
        def do_child_work() -> str:  # type: ignore
            """Child task."""
            pass

        @parent.task("Orchestrate the child")
        def run_parent() -> str:  # type: ignore
            """Parent task."""
            pass

        return parent, child, do_child_work, run_parent

    def test_single_child_report_reaches_parent_log(self):
        parent, child, do_child_work, run_parent = self._setup_parent_child()

        child.llm = Dummy(
            responses=[
                make_response(
                    thinking="doing work",
                    report="Halfway done",
                    code="task_success('child_result')",
                )
            ]
        )
        parent.llm = Dummy(
            responses=[
                make_response(
                    thinking="call the child",
                    code="r = do_child_work()\ntask_success(r)",
                )
            ]
        )

        result = run_parent(session="s1")
        assert result == "child_result"

        parent_state = parent.state("s1")
        parent_events = get_events_from_log(parent_state)
        texts = _output_event_texts(parent_events)
        assert any("[report:child] Halfway done" in t for t in texts), (
            f"parent log did not contain child's report. texts={texts}"
        )

    def test_multiple_children_in_single_block(self):
        """Two sub-task calls in one parent PYTHON block → both reports in parent log."""
        clear_agent_registry()
        state_cfg = connect_state(type="versioned", storage="memory")
        parent = Agent(name="parent", state=state_cfg)
        a = Agent(name="alpha", state=state_cfg)
        b = Agent(name="beta", state=state_cfg)

        @parent.fn(docstring="alpha work")
        @a.task("alpha")
        def alpha_work() -> str:  # type: ignore
            """Alpha."""
            pass

        @parent.fn(docstring="beta work")
        @b.task("beta")
        def beta_work() -> str:  # type: ignore
            """Beta."""
            pass

        @parent.task("Run both")
        def run_both() -> str:  # type: ignore
            """Parent."""
            pass

        a.llm = Dummy(
            responses=[
                make_response(
                    thinking="alpha",
                    report="alpha ran",
                    code="task_success('A')",
                )
            ]
        )
        b.llm = Dummy(
            responses=[
                make_response(
                    thinking="beta",
                    report="beta ran",
                    code="task_success('B')",
                )
            ]
        )
        parent.llm = Dummy(
            responses=[
                make_response(
                    thinking="call both",
                    code="x = alpha_work()\ny = beta_work()\ntask_success(x + y)",
                )
            ]
        )

        result = run_both(session="s2")
        assert result == "AB"

        parent_events = get_events_from_log(parent.state("s2"))
        texts = " ".join(_output_event_texts(parent_events))
        assert "[report:alpha] alpha ran" in texts
        assert "[report:beta] beta ran" in texts

    def test_nested_three_levels_no_leak_to_grandparent(self):
        """A → B → C: C's report reaches B's log, NOT A's log (unless B forwards)."""
        clear_agent_registry()
        state_cfg = connect_state(type="versioned", storage="memory")
        a = Agent(name="a", state=state_cfg)
        b = Agent(name="b", state=state_cfg)
        c = Agent(name="c", state=state_cfg)

        @b.fn(docstring="c work")
        @c.task("c")
        def c_work() -> str:  # type: ignore
            """C."""
            pass

        @a.fn(docstring="b work")
        @b.task("b")
        def b_work() -> str:  # type: ignore
            """B."""
            pass

        @a.task("a")
        def a_work() -> str:  # type: ignore
            """A."""
            pass

        c.llm = Dummy(
            responses=[
                make_response(
                    thinking="c",
                    report="C says hi",
                    code="task_success('c_result')",
                )
            ]
        )
        b.llm = Dummy(
            responses=[
                make_response(
                    thinking="b calls c",
                    code="r = c_work()\ntask_success(r)",
                )
            ]
        )
        a.llm = Dummy(
            responses=[
                make_response(
                    thinking="a calls b",
                    code="r = b_work()\ntask_success(r)",
                )
            ]
        )

        result = a_work(session="nested")
        assert result == "c_result"

        b_texts = " ".join(_output_event_texts(get_events_from_log(b.state("nested"))))
        a_texts = " ".join(_output_event_texts(get_events_from_log(a.state("nested"))))

        # B sees C's report...
        assert "[report:c] C says hi" in b_texts
        # ...but A does NOT (B didn't forward).
        assert "[report:c]" not in a_texts

    def test_forwarding_via_explicit_parent_report(self):
        """If B emits its own REPORT, A sees [report:b] but still not [report:c]."""
        clear_agent_registry()
        state_cfg = connect_state(type="versioned", storage="memory")
        a = Agent(name="a", state=state_cfg)
        b = Agent(name="b", state=state_cfg)
        c = Agent(name="c", state=state_cfg)

        @b.fn(docstring="c work")
        @c.task("c")
        def c_work() -> str:  # type: ignore
            """C."""
            pass

        @a.fn(docstring="b work")
        @b.task("b")
        def b_work() -> str:  # type: ignore
            """B."""
            pass

        @a.task("a")
        def a_work() -> str:  # type: ignore
            """A."""
            pass

        c.llm = Dummy(
            responses=[
                make_response(
                    thinking="c",
                    report="C says hi",
                    code="task_success('c_result')",
                )
            ]
        )
        b.llm = Dummy(
            responses=[
                make_response(
                    thinking="b calls c",
                    report="Summary of C's work",
                    code="r = c_work()\ntask_success(r)",
                )
            ]
        )
        a.llm = Dummy(
            responses=[
                make_response(
                    thinking="a calls b",
                    code="r = b_work()\ntask_success(r)",
                )
            ]
        )

        a_work(session="fwd")

        a_texts = " ".join(_output_event_texts(get_events_from_log(a.state("fwd"))))
        assert "[report:b] Summary of C's work" in a_texts
        # C's own report still doesn't appear in A's log directly.
        assert "[report:c]" not in a_texts

    def test_sub_agent_failure_still_flushes_collected_reports(self):
        """If the sub-agent reports progress then fails, parent still sees the report."""

        parent, child, do_child_work, run_parent = self._setup_parent_child()

        child.llm = Dummy(
            responses=[
                make_response(
                    thinking="reporting then failing",
                    report="Got partial data",
                    code="task_fail('could not finish')",
                )
            ]
        )
        parent.llm = Dummy(
            responses=[
                make_response(
                    thinking="call child",
                    code="try:\n    do_child_work()\nexcept Exception as e:\n    pass\ntask_success('parent done')",
                )
            ]
        )

        run_parent(session="fail")
        parent_events = get_events_from_log(parent.state("fail"))
        texts = " ".join(_output_event_texts(parent_events))
        assert "[report:child] Got partial data" in texts


class TestConsoleHelpers:
    """pprint_events and pprint_tokens render REPORT correctly."""

    def test_pprint_events_shows_report_on_action_event(self):
        from agex.agent.console import pprint_events

        ev = make_action_event(
            agent_name="a",
            thinking="thinking text",
            report="user-visible report text",
            code="pass",
        )
        buf = io.StringIO()
        pprint_events(ev, color="never", show_delta=False, stream=buf)
        out = buf.getvalue()
        assert "Thinking: thinking text" in out
        assert "Report: user-visible report text" in out

    def test_pprint_events_omits_report_line_when_absent(self):
        from agex.agent.console import pprint_events

        ev = make_action_event(agent_name="a", thinking="t", code="pass")
        buf = io.StringIO()
        pprint_events(ev, color="never", show_delta=False, stream=buf)
        out = buf.getvalue()
        assert "Report:" not in out

    def test_pprint_tokens_renders_report_tokens(self):
        from datetime import datetime, timezone

        from agex.agent.console import pprint_tokens
        from agex.llm.core import StreamToken

        buf = io.StringIO()
        start = StreamToken(
            type="report",
            content="Hello user",
            done=False,
            agent_name="a",
            timestamp=datetime.now(timezone.utc),
            start=True,
        )
        more = StreamToken(
            type="report",
            content=", more text",
            done=False,
            agent_name="a",
            timestamp=datetime.now(timezone.utc),
            start=False,
        )
        end = StreamToken(
            type="report",
            content="",
            done=True,
            agent_name="a",
            timestamp=datetime.now(timezone.utc),
        )
        pprint_tokens(start, color="never", stream=buf)
        pprint_tokens(more, color="never", stream=buf)
        pprint_tokens(end, color="never", stream=buf)
        out = buf.getvalue()
        assert "Hello user" in out
        assert ", more text" in out


class TestTopLevelReportNoStdout:
    """Top-level invocations must not write to real stdout."""

    def test_top_level_sub_agent_report_no_stdout_pollution(self):
        clear_agent_registry()
        state_cfg = connect_state(type="versioned", storage="memory")
        parent = Agent(name="parent_top", state=state_cfg)
        child = Agent(name="child_top", state=state_cfg)

        @parent.fn(docstring="child work")
        @child.task("child")
        def child_work() -> str:  # type: ignore
            """Child."""
            pass

        @parent.task("parent")
        def parent_work() -> str:  # type: ignore
            """Parent."""
            pass

        child.llm = Dummy(
            responses=[
                make_response(
                    thinking="c",
                    report="Top-level child reporting in",
                    code="task_success('ok')",
                )
            ]
        )
        parent.llm = Dummy(
            responses=[
                make_response(
                    thinking="p",
                    code="r = child_work()\ntask_success(r)",
                )
            ]
        )

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
            result = parent_work(session="top")

        assert result == "ok"
        assert stdout_buf.getvalue() == "", (
            f"top-level invocation wrote to stdout: {stdout_buf.getvalue()!r}"
        )
        assert stderr_buf.getvalue() == "", (
            f"top-level invocation wrote to stderr: {stderr_buf.getvalue()!r}"
        )
