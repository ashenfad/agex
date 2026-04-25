"""Tests for the tool-use event-log renderer."""

from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
)
from agex.agent.events import (
    ActionEvent,
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
