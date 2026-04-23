"""Tests for the tool-use event-log renderer."""

from agex.agent.datatypes import EditAction, FileAction
from agex.agent.events import (
    ActionEvent,
    ClarifyEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)
from agex.eval.objects import ImageAction
from agex.llm.formats.tool_use import (
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
    render_events_as_tool_use,
)


def _only(blocks, block_type):
    return [b for b in blocks if b.get("type") == block_type]


class TestSingleAction:
    def test_python_action_becomes_assistant_tool_use(self):
        events = [
            TaskStartEvent(
                agent_name="a",
                task_name="t",
                inputs={},
                message="do work",
            ),
            ActionEvent(
                agent_name="a",
                title="Working",
                thinking="step-by-step",
                code="print(1)\ntask_continue()",
            ),
            OutputEvent(agent_name="a", parts=["1"]),
        ]
        msgs = render_events_as_tool_use(events)

        # Shape: [user(task start), assistant(tool_use), user(tool_result)].
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

        # Assistant content is one tool_use for python_action.
        use_blocks = _only(msgs[1]["content"], "tool_use")
        assert len(use_blocks) == 1
        assert use_blocks[0]["name"] == TOOL_PYTHON
        assert use_blocks[0]["input"]["title"] == "Working"
        assert use_blocks[0]["input"]["thinking"] == "step-by-step"
        assert use_blocks[0]["input"]["code"].startswith("print(1)")

        # User content has exactly one tool_result paired with the tool_use.
        results = _only(msgs[2]["content"], "tool_result")
        assert len(results) == 1
        assert results[0]["tool_use_id"] == use_blocks[0]["id"]
        assert "1" in str(results[0]["content"])

    def test_terminal_action(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="Explore",
                thinking="list files",
                terminal="ls -la",
            ),
            OutputEvent(agent_name="a", parts=["file listing"]),
        ]
        msgs = render_events_as_tool_use(events)
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["name"] == TOOL_TERMINAL
        assert use["input"]["commands"] == "ls -la"
        # No code field for terminal.
        assert "code" not in use["input"]

    def test_report_only_included_when_non_empty(self):
        with_report = ActionEvent(
            agent_name="a",
            title="t",
            thinking="T",
            report="working on it",
            code="x",
        )
        without = ActionEvent(agent_name="a", title="t", thinking="T", code="x")
        msgs_with = render_events_as_tool_use([with_report])
        msgs_without = render_events_as_tool_use([without])
        assert (
            _only(msgs_with[0]["content"], "tool_use")[0]["input"]["report"]
            == "working on it"
        )
        assert "report" not in _only(msgs_without[0]["content"], "tool_use")[0]["input"]


class TestFileActions:
    def test_write_file_becomes_additional_tool_use(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
                file_actions=[
                    FileAction(path="/helpers/a.py", content="X = 1"),
                ],
            ),
            OutputEvent(agent_name="a", parts=["ok"]),
        ]
        msgs = render_events_as_tool_use(events)
        assistant_blocks = msgs[0]["content"]
        use_blocks = _only(assistant_blocks, "tool_use")
        assert len(use_blocks) == 2
        assert use_blocks[0]["name"] == TOOL_WRITE_FILE
        assert use_blocks[0]["input"]["path"] == "/helpers/a.py"
        assert use_blocks[0]["input"]["content"] == "X = 1"
        # Default mode is "write" — should be omitted from input.
        assert "mode" not in use_blocks[0]["input"]
        assert use_blocks[1]["name"] == TOOL_PYTHON

        # Next user message must have a tool_result per tool_use, in order.
        results = _only(msgs[1]["content"], "tool_result")
        assert len(results) == 2
        assert results[0]["tool_use_id"] == use_blocks[0]["id"]
        # File tool_result names the tool and what it touched so the
        # LLM can link tool_use ↔ tool_result in plain language.
        assert results[0]["content"] == "write_file: wrote /helpers/a.py"
        assert results[1]["tool_use_id"] == use_blocks[1]["id"]

    def test_append_mode_included(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
                file_actions=[
                    FileAction(path="/x.py", content="more\n", mode="append"),
                ],
            ),
            OutputEvent(agent_name="a", parts=["ok"]),
        ]
        msgs = render_events_as_tool_use(events)
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["input"]["mode"] == "append"

    def test_edit_replace(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
                file_actions=[
                    EditAction(
                        path="/x.py",
                        search="old",
                        content="new",
                        operation="replace",
                    ),
                ],
            ),
            OutputEvent(agent_name="a", parts=["ok"]),
        ]
        msgs = render_events_as_tool_use(events)
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["name"] == TOOL_EDIT_FILE
        assert use["input"]["replace"] == "new"
        assert "insert_after" not in use["input"]

    def test_edit_insert_after(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
                file_actions=[
                    EditAction(
                        path="/x.py",
                        search="anchor",
                        content="added",
                        operation="insert-after",
                        match_all=True,
                    ),
                ],
            ),
            OutputEvent(agent_name="a", parts=["ok"]),
        ]
        msgs = render_events_as_tool_use(events)
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["input"]["insert_after"] == "added"
        assert use["input"]["match_all"] is True


class TestObservationPairing:
    def test_success_event_fills_main_tool_result(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_success(42)",
            ),
            SuccessEvent(agent_name="a", result=42),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert len(results) == 1
        # Result is rendered — should contain "42" somewhere.
        assert "42" in str(results[0]["content"])

    def test_fail_event_names_tool_and_carries_message(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_fail('bad creds')",
            ),
            FailEvent(agent_name="a", message="bad creds"),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert len(results) == 1
        # Framing includes the tool name AND surfaces the fail message
        # so the LLM can read what happened on its previous turn.
        assert results[0]["content"] == "python_action: task_fail: bad creds"

    def test_clarify_event_names_tool_and_carries_message(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_clarify('which dataset?')",
            ),
            ClarifyEvent(agent_name="a", message="which dataset?"),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert results[0]["content"] == "python_action: task_clarify: which dataset?"

    def test_no_observation_synthesizes_named_placeholder(self):
        """Trailing ActionEvent with no follow-up still needs a tool_result
        for a well-formed tool_use pairing.  The placeholder names the
        tool so the LLM sees the linkage."""
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
            ),
        ]
        msgs = render_events_as_tool_use(events)
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        results = _only(msgs[1]["content"], "tool_result")
        assert results[0]["content"] == "python_action: (no observation)"

    def test_python_action_output_prefixed_with_tool_name(self):
        """OutputEvent text (stdout etc.) should land in the tool_result
        with a tool-name prefix so the LLM can read it as "output from
        my python_action."""
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="print('hello'); task_continue()",
            ),
            OutputEvent(agent_name="a", parts=["hello"]),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.startswith("python_action: output")
        assert "hello" in content

    def test_terminal_action_output_prefixed_with_tool_name(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                terminal="ls -la",
            ),
            OutputEvent(agent_name="a", parts=["file1\nfile2"]),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.startswith("terminal_action: output")
        assert "file1" in content

    def test_print_large_non_string_arg_not_truncated(self):
        """Regression: ``print(big_list)`` previously rendered through
        ``render_value`` (default 2048-char budget) which silently
        chopped the LLM's view of large printed values.  The studio
        UI used ``str()`` directly and showed everything, so users saw
        the full output in history but the agent's next-turn observation
        was truncated.  Must use ``str()`` semantics, matching ``print()``."""
        from agex.eval.objects import PrintAction

        # Build a list whose str() is well over 2048 chars.
        big_list = [{"i": i, "label": f"action_{i}", "ok": True} for i in range(200)]
        big_str = str(big_list)
        assert len(big_str) > 4000  # sanity: definitely over the old budget

        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="print(test_app(...))",
            ),
            OutputEvent(agent_name="a", parts=[PrintAction((big_list,))]),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        # The full str() of the list should appear, including the very
        # last entry — the failure mode was "trailing items lost."
        assert "action_199" in content
        assert "action_0" in content

    def test_print_string_args_unwrapped_no_repr_quotes(self):
        """Regression: previously ``PrintAction(('hello',))`` rendered as
        ``'hello'`` (repr-wrapped) in tool_result text, producing odd
        stray quotes.  Must appear as ``hello``."""
        from agex.eval.objects import PrintAction

        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="print('hi')",
            ),
            OutputEvent(agent_name="a", parts=[PrintAction(("hi",))]),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.endswith("hi")
        # No repr-style wrapping.
        assert "'hi'" not in content

    def test_edit_file_result_names_operation_and_path(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="pass",
                file_actions=[
                    EditAction(
                        path="/x.py",
                        search="X",
                        content="Y",
                        operation="insert-after",
                        match_all=True,
                    )
                ],
            ),
            OutputEvent(agent_name="a", parts=["ok"]),
        ]
        msgs = render_events_as_tool_use(events)
        file_result = _only(msgs[-1]["content"], "tool_result")[0]
        assert file_result["content"] == (
            "edit_file: insert-after applied to /x.py (match_all)"
        )

    def test_success_event_result_prefixed_with_tool_name(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_success(42)",
            ),
            SuccessEvent(agent_name="a", result=42),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.startswith("python_action: task_success returned")
        assert "42" in content

    def test_output_with_image_uses_content_parts(self):
        from PIL import Image

        img = Image.new("RGB", (4, 4), color="red")
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="plot_something()",
            ),
            OutputEvent(
                agent_name="a",
                parts=["plot below", ImageAction(image=img)],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        result = _only(msgs[-1]["content"], "tool_result")[0]
        assert isinstance(result["content"], list)
        types = [p["type"] for p in result["content"]]
        assert "text" in types
        assert "image" in types


class TestMultiTurn:
    def test_multiple_tasks_flushed_correctly(self):
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="task 1"),
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_success(1)",
            ),
            SuccessEvent(agent_name="a", result=1),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="task 2"),
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="task_success(2)",
            ),
            SuccessEvent(agent_name="a", result=2),
        ]
        msgs = render_events_as_tool_use(events)
        roles = [m["role"] for m in msgs]
        # Expect: user, assistant, user, assistant, user
        # (task 2's start text attaches to the preceding tool_result user msg.)
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        # Second user message (post-task-1) contains both the tool_result
        # AND task 2's start text.
        second_user = msgs[2]["content"]
        has_text = any(b.get("type") == "text" for b in second_user)
        has_result = any(b.get("type") == "tool_result" for b in second_user)
        assert has_text and has_result

    def test_tool_use_ids_are_unique(self):
        events = [
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="x",
                file_actions=[
                    FileAction(path="/a.py", content="A"),
                    FileAction(path="/b.py", content="B"),
                ],
            ),
            OutputEvent(agent_name="a", parts=["done"]),
            ActionEvent(
                agent_name="a",
                title="t",
                thinking="T",
                code="y",
            ),
            SuccessEvent(agent_name="a", result=None),
        ]
        msgs = render_events_as_tool_use(events)
        ids: list[str] = []
        for m in msgs:
            if isinstance(m["content"], list):
                for b in m["content"]:
                    if b.get("type") == "tool_use":
                        ids.append(b["id"])
        assert len(ids) == len(set(ids))


class TestEmpty:
    def test_no_events(self):
        assert render_events_as_tool_use([]) == []

    def test_only_task_start(self):
        events = [
            TaskStartEvent(agent_name="a", task_name="t", inputs={}, message="go")
        ]
        msgs = render_events_as_tool_use(events)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Content is a list of text parts.
        assert msgs[0]["content"][0]["type"] == "text"
