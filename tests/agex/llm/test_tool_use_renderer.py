"""Tests for the tool-use event-log renderer."""

from agex.agent.chapter import CHAPTER_TASK
from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
)
from agex.agent.events import (
    ActionEvent,
    ChapterEvent,
    ClarifyEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.objects import ImageAction, PrintAction
from agex.llm.formats.tool_use import (
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
    render_events_as_tool_use,
)


def _only(blocks, block_type):
    return [b for b in blocks if b.get("type") == block_type]


def _python_action(
    code: str = "pass",
    *,
    title: str | None = "Working",
    thinking: str | None = "step-by-step",
) -> ActionEvent:
    return ActionEvent(
        agent_name="a",
        emissions=[PythonEmission(code=code, title=title, thinking=thinking)],
    )


def _terminal_action(
    commands: str = "ls",
    *,
    title: str | None = "Explore",
    thinking: str | None = "list files",
) -> ActionEvent:
    return ActionEvent(
        agent_name="a",
        emissions=[
            TerminalEmission(commands=commands, title=title, thinking=thinking),
        ],
    )


class TestSingleAction:
    def test_python_action_becomes_assistant_tool_use(self):
        events = [
            TaskStartEvent(
                agent_name="a",
                task_name="t",
                inputs={},
                message="do work",
            ),
            _python_action(code="print(1)"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("1",), emission_id="em_1_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)

        # Shape: [user(task start), assistant(tool_use), user(tool_result)].
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]

        use_blocks = _only(msgs[1]["content"], "tool_use")
        assert len(use_blocks) == 1
        assert use_blocks[0]["name"] == TOOL_PYTHON
        assert use_blocks[0]["input"]["title"] == "Working"
        assert use_blocks[0]["input"]["thinking"] == "step-by-step"
        assert use_blocks[0]["input"]["code"].startswith("print(1)")

        results = _only(msgs[2]["content"], "tool_result")
        assert len(results) == 1
        assert results[0]["tool_use_id"] == use_blocks[0]["id"]
        assert "1" in str(results[0]["content"])

    def test_terminal_action(self):
        events = [
            _terminal_action(commands="ls -la"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("file listing",), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["name"] == TOOL_TERMINAL
        assert use["input"]["commands"] == "ls -la"
        assert "code" not in use["input"]

    def test_text_emission_becomes_assistant_text_block(self):
        """A TextEmission (the new user-facing prose channel) renders as
        a plain text block in the assistant message — not as an input
        field on the tool_use.
        """
        event = ActionEvent(
            agent_name="a",
            emissions=[
                TextEmission(text="working on it"),
                PythonEmission(code="x", title="Working", thinking="T"),
            ],
        )
        msgs = render_events_as_tool_use([event])
        texts = _only(msgs[0]["content"], "text")
        assert [t["text"] for t in texts] == ["working on it"]
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert "report" not in use["input"]


class TestFileEmissions:
    def test_write_file_becomes_additional_tool_use(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileWriteEmission(path="/helpers/a.py", content="X = 1"),
                PythonEmission(code="pass", title="t", thinking="T"),
            ],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("ok",), emission_id="em_0_1")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        assistant_blocks = msgs[0]["content"]
        use_blocks = _only(assistant_blocks, "tool_use")
        assert len(use_blocks) == 2
        assert use_blocks[0]["name"] == TOOL_WRITE_FILE
        assert use_blocks[0]["input"]["path"] == "/helpers/a.py"
        assert use_blocks[0]["input"]["content"] == "X = 1"
        assert "mode" not in use_blocks[0]["input"]
        assert use_blocks[1]["name"] == TOOL_PYTHON

        results = _only(msgs[1]["content"], "tool_result")
        assert len(results) == 2
        assert results[0]["tool_use_id"] == use_blocks[0]["id"]
        assert results[0]["content"] == "write_file: wrote /helpers/a.py"
        assert results[1]["tool_use_id"] == use_blocks[1]["id"]

    def test_append_mode_included(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileWriteEmission(path="/x.py", content="more\n", mode="append"),
                PythonEmission(code="pass", title="t", thinking="T"),
            ],
        )
        msgs = render_events_as_tool_use([action])
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["input"]["mode"] == "append"

    def test_edit_replace(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileEditEmission(
                    path="/x.py",
                    search="old",
                    content="new",
                ),
                PythonEmission(code="pass", title="t", thinking="T"),
            ],
        )
        msgs = render_events_as_tool_use([action])
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["name"] == TOOL_EDIT_FILE
        assert use["input"]["replace"] == "new"
        assert "insert_after" not in use["input"]

    def test_edit_match_all_flag_round_trips(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileEditEmission(
                    path="/x.py",
                    search="anchor",
                    content="anchor + extra",
                    match_all=True,
                ),
                PythonEmission(code="pass", title="t", thinking="T"),
            ],
        )
        msgs = render_events_as_tool_use([action])
        use = _only(msgs[0]["content"], "tool_use")[0]
        assert use["input"]["replace"] == "anchor + extra"
        assert use["input"]["match_all"] is True


class TestObservationPairing:
    def test_success_event_fills_python_tool_result(self):
        events = [
            _python_action(code="task_success(42)"),
            SuccessEvent(agent_name="a", result=42),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert len(results) == 1
        assert "42" in str(results[0]["content"])

    def test_fail_event_names_tool_and_carries_message(self):
        events = [
            _python_action(code="task_fail('bad creds')"),
            FailEvent(agent_name="a", message="bad creds"),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert len(results) == 1
        assert results[0]["content"] == "python_action: task_fail: bad creds"

    def test_clarify_event_names_tool_and_carries_message(self):
        events = [
            _python_action(code="task_clarify('which dataset?')"),
            ClarifyEvent(agent_name="a", message="which dataset?"),
        ]
        msgs = render_events_as_tool_use(events)
        results = _only(msgs[-1]["content"], "tool_result")
        assert results[0]["content"] == "python_action: task_clarify: which dataset?"

    def test_no_observation_synthesizes_named_placeholder(self):
        events = [_python_action(code="pass")]
        msgs = render_events_as_tool_use(events)
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        results = _only(msgs[1]["content"], "tool_result")
        assert results[0]["content"] == "python_action: (no observation)"

    def test_python_action_output_prefixed_with_tool_name(self):
        events = [
            _python_action(code="print('hello')"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("hello",), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.startswith("python_action: output")
        assert "hello" in content

    def test_terminal_action_output_prefixed_with_tool_name(self):
        events = [
            _terminal_action(commands="ls -la"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("file1\nfile2",), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.startswith("terminal_action: output")
        assert "file1" in content

    def test_multiple_output_events_aggregate_into_one_tool_result(self):
        """A Python turn emits one OutputEvent per print() call; all of
        them pair to the emission's single tool_result."""
        events = [
            _python_action(code="print(a); print(b); print(c)"),
            OutputEvent(
                agent_name="a",
                parts=[
                    PrintAction(args=("[read .hud] hp 50/50",), emission_id="em_0_0")
                ],
            ),
            OutputEvent(
                agent_name="a",
                parts=[
                    PrintAction(args=("[eval] restart btn ok",), emission_id="em_0_0")
                ],
            ),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("[eval] 2",), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        tool_results = _only(msgs[-1]["content"], "tool_result")
        assert len(tool_results) == 1
        content = tool_results[0]["content"]
        assert "hp 50/50" in content
        assert "restart btn ok" in content
        assert "[eval] 2" in content
        assert content.count("python_action: output") == 1

    def test_prints_plus_task_success_both_land_in_tool_result(self):
        events = [
            _python_action(code="print('done!'); task_success(42)"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("done!",), emission_id="em_0_0")],
            ),
            SuccessEvent(agent_name="a", result=42),
        ]
        msgs = render_events_as_tool_use(events)
        tool_results = _only(msgs[-1]["content"], "tool_result")
        assert len(tool_results) == 1
        content = tool_results[0]["content"]
        assert "done!" in content
        assert "task_success returned" in content
        assert "42" in content

    def test_multiple_view_images_all_appear_in_tool_result(self):
        from PIL import Image

        img1 = Image.new("RGB", (4, 4), color="red")
        img2 = Image.new("RGB", (4, 4), color="green")
        events = [
            _python_action(code="view_image(a); view_image(b)"),
            OutputEvent(
                agent_name="a",
                parts=[ImageAction(image=img1, emission_id="em_0_0")],
            ),
            OutputEvent(
                agent_name="a",
                parts=[ImageAction(image=img2, emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        tool_results = _only(msgs[-1]["content"], "tool_result")
        assert len(tool_results) == 1
        content = tool_results[0]["content"]
        assert isinstance(content, list)
        image_blocks = [b for b in content if b.get("type") == "image"]
        assert len(image_blocks) == 2

    def test_tool_result_precedes_text_parts_in_user_message(self):
        """Anthropic requires tool_result blocks before free-form text
        in a user message."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="task 1"),
            _python_action(code="print('x')"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("x",), emission_id="em_1_0")],
            ),
            TaskStartEvent(
                agent_name="a", task_name="t2", inputs={}, message="next task"
            ),
            _python_action(code="task_success(1)"),
            SuccessEvent(agent_name="a", result=1),
        ]
        msgs = render_events_as_tool_use(events)
        # The user message bridging task 1 → task 2 has tool_result
        # before text.
        for msg in msgs:
            if msg["role"] != "user":
                continue
            types = [b.get("type") for b in msg["content"]]
            if "tool_result" in types and "text" in types:
                tool_result_idx = types.index("tool_result")
                text_idx = types.index("text")
                assert tool_result_idx < text_idx
                return
        raise AssertionError("no bridging user message found")

    def test_print_large_non_string_arg_not_truncated(self):
        big_list = [{"i": i, "label": f"action_{i}", "ok": True} for i in range(200)]
        big_str = str(big_list)
        assert len(big_str) > 4000

        events = [
            _python_action(code="print(test_app(...))"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=(big_list,), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert "action_199" in content
        assert "action_0" in content

    def test_print_string_args_unwrapped_no_repr_quotes(self):
        events = [
            _python_action(code="print('hi')"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("hi",), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert content.endswith("hi")
        assert "'hi'" not in content

    def test_print_huge_string_arg_is_truncated(self):
        big_blob = "A" * 200_000
        events = [
            _python_action(code="print(big_blob)"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=(big_blob,), emission_id="em_0_0")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        content = _only(msgs[-1]["content"], "tool_result")[0]["content"]
        assert len(content) < 100_000
        assert "truncated" in content
        assert "200000" in content

    def test_edit_file_result_names_operation_and_path(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileEditEmission(
                    path="/x.py",
                    search="X",
                    content="Y",
                    match_all=True,
                ),
                PythonEmission(code="pass", title="t", thinking="T"),
            ],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("ok",), emission_id="em_0_1")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        file_result = _only(msgs[-1]["content"], "tool_result")[0]
        assert file_result["content"] == (
            "edit_file: replace applied to /x.py (match_all)"
        )

    def test_file_edit_error_surfaced_instead_of_synth(self):
        """When an edit_file fails (search not matched), the loop
        emits an error OutputEvent stamped with the emission_id.  The
        renderer must surface that error in the tool_result rather
        than the synthesized "edit_file: replace applied" success
        line — otherwise the agent believes the edit succeeded and
        proceeds with broken state."""
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileEditEmission(
                    path="/x.py",
                    search="nonexistent",
                    content="new",
                ),
            ],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[
                    PrintAction(
                        args=("💥 ResponseParseError: EDIT search not found in /x.py",),
                        emission_id="em_0_0",
                    )
                ],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        result = _only(msgs[-1]["content"], "tool_result")[0]
        assert "applied to /x.py" not in result["content"]
        assert "EDIT search not found" in result["content"]

    def test_file_write_error_surfaced_instead_of_synth(self):
        action = ActionEvent(
            agent_name="a",
            emissions=[FileWriteEmission(path="/x.py", content="X")],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[
                    PrintAction(
                        args=("💥 OSError: read-only filesystem",),
                        emission_id="em_0_0",
                    )
                ],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        result = _only(msgs[-1]["content"], "tool_result")[0]
        assert "wrote /x.py" not in result["content"]
        assert "read-only filesystem" in result["content"]

    def test_success_event_result_prefixed_with_tool_name(self):
        events = [
            _python_action(code="task_success(42)"),
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
            _python_action(code="plot_something()"),
            OutputEvent(
                agent_name="a",
                parts=[
                    PrintAction(args=("plot below",), emission_id="em_0_0"),
                    ImageAction(image=img, emission_id="em_0_0"),
                ],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        result = _only(msgs[-1]["content"], "tool_result")[0]
        assert isinstance(result["content"], list)
        types = [p["type"] for p in result["content"]]
        assert "text" in types
        assert "image" in types


class TestMultiEmissionTurn:
    def test_file_plus_python_emissions_produce_two_tool_results(self):
        """One turn with [FileWrite, Python] produces two tool_use blocks
        in the assistant message and two tool_results in the next user
        message."""
        action = ActionEvent(
            agent_name="a",
            emissions=[
                FileWriteEmission(path="/helpers/x.py", content="X = 1"),
                PythonEmission(
                    code="from helpers.x import X; print(X)",
                    title="Try import",
                    thinking="import the module we just wrote",
                ),
            ],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("1",), emission_id="em_0_1")],
            ),
        ]
        msgs = render_events_as_tool_use(events)

        assistant_blocks = msgs[0]["content"]
        use_blocks = _only(assistant_blocks, "tool_use")
        assert [b["name"] for b in use_blocks] == [TOOL_WRITE_FILE, TOOL_PYTHON]

        tool_results = _only(msgs[1]["content"], "tool_result")
        assert len(tool_results) == 2
        assert tool_results[0]["content"] == "write_file: wrote /helpers/x.py"
        assert "1" in str(tool_results[1]["content"])

    def test_prints_route_to_their_originating_emission(self):
        """Two PythonEmissions in one turn; each emission's prints should
        end up under *its* tool_result via emission_id pairing."""
        action = ActionEvent(
            agent_name="a",
            emissions=[
                PythonEmission(code="print('one')", title="first", thinking="T"),
                PythonEmission(code="print('two')", title="second", thinking="T"),
            ],
        )
        events = [
            action,
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("one",), emission_id="em_0_0")],
            ),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("two",), emission_id="em_0_1")],
            ),
        ]
        msgs = render_events_as_tool_use(events)
        tool_results = _only(msgs[1]["content"], "tool_result")
        assert len(tool_results) == 2
        assert "one" in str(tool_results[0]["content"])
        assert "two" in str(tool_results[1]["content"])
        # Cross-contamination check: first result should NOT contain the
        # second emission's output.
        assert "two" not in str(tool_results[0]["content"])


class TestMultiTurn:
    def test_multiple_tasks_flushed_correctly(self):
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="task 1"),
            _python_action(code="task_success(1)"),
            SuccessEvent(agent_name="a", result=1),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="task 2"),
            _python_action(code="task_success(2)"),
            SuccessEvent(agent_name="a", result=2),
        ]
        msgs = render_events_as_tool_use(events)
        roles = [m["role"] for m in msgs]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        second_user = msgs[2]["content"]
        has_text = any(b.get("type") == "text" for b in second_user)
        has_result = any(b.get("type") == "tool_result" for b in second_user)
        assert has_text and has_result

    def test_tool_use_ids_are_unique(self):
        events = [
            ActionEvent(
                agent_name="a",
                emissions=[
                    FileWriteEmission(path="/a.py", content="A"),
                    FileWriteEmission(path="/b.py", content="B"),
                    PythonEmission(code="x", title="t", thinking="T"),
                ],
            ),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("done",), emission_id="em_0_2")],
            ),
            _python_action(code="y"),
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


class TestSystemNoteFraming:
    """SystemNoteEvent messages render with a ``[system]`` prefix so
    the model can disambiguate framework telemetry from user speech in
    user-role messages.  The persisted event keeps its raw message
    (no presentation concerns leak into the log)."""

    def test_system_note_gets_prefix(self):
        events = [
            _python_action(code="x"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("done",), emission_id="em_0_0")],
            ),
            SystemNoteEvent(
                agent_name="System",
                message="✓ write_file: /helpers/utils.py",
            ),
        ]
        msgs = render_events_as_tool_use(events)
        # System note lands as trailing text in the user message that
        # carries the tool_result for the prior python_action.
        assert len(msgs) == 2  # assistant turn + user turn
        user_msg = msgs[1]
        assert user_msg["role"] == "user"
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        assert any(
            b["text"] == "[system] ✓ write_file: /helpers/utils.py" for b in text_blocks
        )

    def test_persisted_event_message_is_unchanged(self):
        """The prefix lives in the renderer, not on the event — the
        log entry stays clean so the wrapping convention can evolve
        without migrating historical data."""
        event = SystemNoteEvent(agent_name="System", message="raw note")
        # Event body has no prefix.
        assert event.message == "raw note"
        # Renderer adds the prefix on the way out.
        msgs = render_events_as_tool_use([_python_action(code="x"), event])
        user_msg = msgs[1]
        text_blocks = [b for b in user_msg["content"] if b.get("type") == "text"]
        assert any(b["text"] == "[system] raw note" for b in text_blocks)

    def test_multiple_system_notes_each_get_prefix(self):
        events = [
            _python_action(code="x"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("ok",), emission_id="em_0_0")],
            ),
            SystemNoteEvent(agent_name="System", message="first"),
            SystemNoteEvent(agent_name="System", message="second"),
        ]
        msgs = render_events_as_tool_use(events)
        user_msg = msgs[1]
        texts = [b["text"] for b in user_msg["content"] if b.get("type") == "text"]
        assert "[system] first" in texts
        assert "[system] second" in texts


class TestChapterScopeFiltering:
    """Filter A: closed ``__chapter__`` task scopes are skipped during
    rendering so the agent's context shows the resulting ChapterEvent
    summary without the chapter task's own bookkeeping (which carries
    the same summary text in its ``task_success([Chapter(...)])`` code).
    """

    def test_closed_chapter_scope_is_skipped(self):
        """The chapter task's TaskStart, action carrying the
        ``task_success([Chapter(...)])``, and closing Success do not
        appear in the rendered messages — only the resulting
        ChapterEvent does."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="real_work()"),
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("done",), emission_id="em_1_0")],
            ),
            SuccessEvent(agent_name="a", result="done"),
            # Chapter task scope — should be filtered out
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message="chapter primer goes here",
            ),
            _python_action(
                code='task_success([Chapter(start=1, end=1, name="P1", message="full summary")])'
            ),
            SuccessEvent(agent_name="a", result=[]),
            # The resulting ChapterEvent
            ChapterEvent(agent_name="a", name="P1", message="full summary"),
            # Next parent task
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
            _python_action(code="more_work()"),
            SuccessEvent(agent_name="a", result="r2"),
        ]
        msgs = render_events_as_tool_use(events)
        # Stringify the whole render — text blocks and tool_use inputs
        # — so we can spot bookkeeping leaks regardless of where they
        # land (assistant text turns, user text, or tool_use input).
        rendered_str = repr(msgs)
        # The chapter task's primer / bookkeeping must not appear.
        assert "chapter primer goes here" not in rendered_str
        assert CHAPTER_TASK not in rendered_str
        # The chapter task's emitted summary code shouldn't appear
        # either — the ChapterEvent already carries the summary.
        assert "Chapter(start=1, end=1" not in rendered_str
        # The ChapterEvent summary IS rendered.
        assert "P1" in rendered_str
        assert "full summary" in rendered_str
        # And t2's real action code still renders (in a tool_use block).
        assert "more_work()" in rendered_str

    def test_open_chapter_scope_is_visible(self):
        """While the chapter task is still running (no terminator yet),
        its scope is OPEN — its TaskStart and any prior turns must
        remain visible to its own loop's render call."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="x"),
            SuccessEvent(agent_name="a", result="r"),
            # Chapter task starts — no terminator yet
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message="chapter primer goes here",
            ),
            _python_action(code="/* turn 1: still picking */"),
        ]
        msgs = render_events_as_tool_use(events)
        flat = "\n".join(
            (b.get("text", "") if isinstance(b, dict) else "")
            for m in msgs
            for b in (
                m["content"]
                if isinstance(m["content"], list)
                else [{"type": "text", "text": m["content"]}]
            )
        )
        # The chapter task's prompt + turn 1 are visible — the open
        # scope is exempt from Filter A so the chapter task can see
        # its own conversation history when it runs turn 2.
        assert "chapter primer goes here" in flat

    def test_multi_turn_chapter_task_sees_its_own_prior_turns(self):
        """Regression for the agex-ts case at chaptering.test.ts:293.

        When a chapter task emits a non-terminal action on turn 1 and
        runs a second turn, turn 2's render must include the chapter
        task's own turn-1 action.  Filter A's open-scope contract: an
        unclosed __chapter__ scope is NOT skipped, so the chapter task
        sees its own conversation history when its loop calls
        ``render_events_as_tool_use``.

        If Filter A wrongly filtered the open scope, the chapter task's
        LLM call on turn 2 would see only the parent's events — its
        turn-1 thinking, code, and any output would vanish, breaking
        provider-side tool_use/tool_result pairing.
        """
        events = [
            # Parent task ran and finished.
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="parent_action()"),
            SuccessEvent(agent_name="a", result="r1"),
            # Chapter task starts (no terminator yet — open scope).
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message="chapter task prompt",
            ),
            # Turn 1 action — *non-terminal*, e.g. the agent narrating
            # its plan before emitting Chapter() instances.
            ActionEvent(
                agent_name="a",
                emissions=[
                    PythonEmission(
                        code="# turn 1: thinking about which to fold",
                        title="Plan",
                        thinking="I'll fold t1.",
                    )
                ],
            ),
            # OutputEvent for turn 1 — must route to turn 1's tool_use.
            OutputEvent(
                agent_name="a",
                parts=[PrintAction(args=("scoping",), emission_id="em_4_0")],
            ),
            # (No SuccessEvent yet — chapter task is mid-flight, about
            #  to emit turn 2.)
        ]
        msgs = render_events_as_tool_use(events)
        rendered = repr(msgs)
        # Chapter task's turn-1 thinking/title/code must be visible —
        # this is what proves Filter A doesn't strip the open scope.
        assert "turn 1: thinking about which to fold" in rendered
        assert "Plan" in rendered
        assert "I'll fold t1." in rendered
        # Its prompt is also visible (the TaskStartEvent message).
        assert "chapter task prompt" in rendered
        # Tool_use / tool_result pairing for turn 1 is intact: there
        # must be a tool_use block with id "em_4_0" AND a tool_result
        # block referencing that id (carrying the "scoping" output).
        rendered_lower = rendered
        assert "em_4_0" in rendered_lower
        assert "scoping" in rendered_lower

    def test_filter_a_does_not_inflate_token_count(self):
        """Token-count regression: the rendered output of a log that
        contains a closed __chapter__ scope is approximately the same
        size as the same log with the bookkeeping events stripped.

        The whole point of Filter A is "the chapter task's bookkeeping
        is invisible to the parent's next render."  This pins down
        that property at the token level — a future change that
        leaks chapter bookkeeping into the render would inflate the
        count and fail this test.
        """
        from agex.tokenizers import get_tokenizer

        baseline = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="real_work()"),
            SuccessEvent(agent_name="a", result="r1"),
            ChapterEvent(agent_name="a", name="P1", message="full summary text"),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
            _python_action(code="more_work()"),
            SuccessEvent(agent_name="a", result="r2"),
        ]
        # Same log, but with a fully-formed chapter task scope inserted
        # before the ChapterEvent (matching what the runtime actually
        # produces — the bookkeeping is what Filter A is meant to hide).
        with_bookkeeping = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="real_work()"),
            SuccessEvent(agent_name="a", result="r1"),
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={"event_index": "[1] task t1"},
                message=(
                    "this is a long chapter primer with many words so the "
                    "token cost of leaking it would be obvious if Filter A "
                    "ever silently regressed"
                ),
            ),
            _python_action(
                code='task_success([Chapter(start=1, end=1, name="P1", message="full summary text")])'
            ),
            SuccessEvent(agent_name="a", result=[]),
            ChapterEvent(agent_name="a", name="P1", message="full summary text"),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
            _python_action(code="more_work()"),
            SuccessEvent(agent_name="a", result="r2"),
        ]

        def _token_count(events):
            tokenizer = get_tokenizer("gpt-4")
            messages = render_events_as_tool_use(events)
            total = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total += len(tokenizer.encode(content))
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text") or ""
                            if text:
                                total += len(tokenizer.encode(text))
                            # Tool-use blocks carry their input as a
                            # dict — count its serialized form so leaks
                            # of chapter-task code into a tool_use block
                            # would be detected too.
                            if part.get("type") == "tool_use":
                                total += len(
                                    tokenizer.encode(repr(part.get("input", {})))
                                )
            return total

        baseline_tokens = _token_count(baseline)
        with_book_tokens = _token_count(with_bookkeeping)

        # Allow a small tolerance for incidental positional differences
        # (e.g. tool_use_id formatting).  Anything bigger means the
        # chapter task's primer / code is leaking into the render.
        assert abs(with_book_tokens - baseline_tokens) <= 5, (
            f"Filter A regression: baseline={baseline_tokens}, "
            f"with_bookkeeping={with_book_tokens} (delta should be ~0)"
        )

    def test_multiple_consecutive_closed_chapter_scopes(self):
        """Two closed chapter scopes back-to-back are both filtered
        — exercises stack discipline across consecutive scopes."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            SuccessEvent(agent_name="a", result="r1"),
            # First chapter task scope.
            TaskStartEvent(
                agent_name="a", task_name=CHAPTER_TASK, inputs={}, message="primer-1"
            ),
            _python_action(code="task_success([])"),
            SuccessEvent(agent_name="a", result=[]),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
            SuccessEvent(agent_name="a", result="r2"),
            # Second chapter task scope.
            TaskStartEvent(
                agent_name="a", task_name=CHAPTER_TASK, inputs={}, message="primer-2"
            ),
            _python_action(code="task_success([])"),
            SuccessEvent(agent_name="a", result=[]),
            # Third real task.
            TaskStartEvent(agent_name="a", task_name="t3", inputs={}, message="t3"),
        ]
        rendered = repr(render_events_as_tool_use(events))
        assert "primer-1" not in rendered
        assert "primer-2" not in rendered
        assert CHAPTER_TASK not in rendered
        # All three real task starts remain visible.
        assert "t1" in rendered
        assert "t2" in rendered
        assert "t3" in rendered

    def test_non_chapter_subtask_inside_chapter_scope_is_filtered(self):
        """A sub-task running inside a chapter scope is also filtered
        — the chapter-scope filter's stack discipline keeps the chapter
        frame on the stack across the inner task's open/close, so the
        whole chapter scope (including the inner task's events) is
        skipped."""
        events = [
            TaskStartEvent(
                agent_name="a", task_name="parent", inputs={}, message="parent"
            ),
            SuccessEvent(agent_name="a", result="r"),
            # Chapter task opens.
            TaskStartEvent(
                agent_name="a", task_name=CHAPTER_TASK, inputs={}, message="ch-primer"
            ),
            # Nested non-chapter task inside the chapter scope (e.g. a
            # tool a chapter task ran).  Its own taskStart and success
            # do NOT close the chapter frame.
            TaskStartEvent(
                agent_name="a", task_name="inner", inputs={}, message="inner-msg"
            ),
            _python_action(code="inner_action()"),
            SuccessEvent(agent_name="a", result="inner-r"),
            # Chapter task continues, closes.
            _python_action(code="task_success([])"),
            SuccessEvent(agent_name="a", result=[]),
            # Real next task.
            TaskStartEvent(agent_name="a", task_name="next", inputs={}, message="next"),
        ]
        rendered = repr(render_events_as_tool_use(events))
        # All chapter-scope content invisible — including the inner task.
        assert "ch-primer" not in rendered
        assert "inner-msg" not in rendered
        assert "inner_action()" not in rendered
        assert "inner-r" not in rendered
        assert CHAPTER_TASK not in rendered
        # Real surrounding tasks remain.
        assert "parent" in rendered
        assert "next" in rendered

    def test_open_chapter_scope_mid_log_remains_visible(self):
        """Defensive: an open chapter scope with events after it (e.g.
        a chapter task that crashed mid-flight, leaving its taskStart
        in the log without a terminator while the parent continued)
        is NOT filtered by Filter A.

        This is a degenerate state — in normal flow the chapter task
        either closes cleanly or its enclosing task ends.  Filter A's
        contract is "skip *closed* scopes only," so the open scope
        renders.  The test pins down the behavior so a future change
        that auto-closes orphaned scopes is a deliberate decision, not
        a silent one.
        """
        events = [
            TaskStartEvent(
                agent_name="a", task_name="parent", inputs={}, message="parent"
            ),
            _python_action(code="x"),
            SuccessEvent(agent_name="a", result="r"),
            # Open chapter scope (no terminator) — orphaned.
            TaskStartEvent(
                agent_name="a",
                task_name=CHAPTER_TASK,
                inputs={},
                message="orphan-primer",
            ),
            _python_action(code="orphan_code"),
            # No closing terminator for the chapter task.
            # Parent continues with another task afterward.
            TaskStartEvent(agent_name="a", task_name="next", inputs={}, message="next"),
        ]
        rendered = repr(render_events_as_tool_use(events))
        # Open scope remains visible (Filter A only skips closed scopes).
        assert "orphan-primer" in rendered
        assert "orphan_code" in rendered

    def test_task_numbering_skips_filtered_starts(self):
        """A filtered chapter task's TaskStart does not bump the
        ``[N]`` counter — t2 stays ``[2]`` (matching what
        ``build_boundary_index`` would produce)."""
        events = [
            TaskStartEvent(agent_name="a", task_name="t1", inputs={}, message="t1"),
            _python_action(code="x"),
            SuccessEvent(agent_name="a", result="r1"),
            TaskStartEvent(
                agent_name="a", task_name=CHAPTER_TASK, inputs={}, message="chapter"
            ),
            _python_action(code="task_success([])"),
            SuccessEvent(agent_name="a", result=[]),
            TaskStartEvent(agent_name="a", task_name="t2", inputs={}, message="t2"),
        ]
        msgs = render_events_as_tool_use(events)
        flat = "\n".join(
            (b.get("text", "") if isinstance(b, dict) else "")
            for m in msgs
            for b in (
                m["content"]
                if isinstance(m["content"], list)
                else [{"type": "text", "text": m["content"]}]
            )
        )
        # t1 is [1], t2 is [2] — chapter task does NOT take slot [2].
        assert "[1] t1" in flat
        assert "[2] t2" in flat
        assert "[3]" not in flat
