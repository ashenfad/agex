"""agex's stored runs, read into nontainer's trace shape: ``AgexTraces``."""

import pytest
from nontainer import Store
from nontainer.conformance.corpus import calls, says, writes
from nontainer.traces import TraceEvent, TraceTokens

from agex import Agent
from agex.agent import HARNESS
from agex.providers.scripted import ScriptedProvider
from agex.record import (
    Image,
    Message,
    Run,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Usage,
    dump_run,
)
from agex.traces import AgexTraces


def read(*messages, **kw):
    kw.setdefault("run_id", "r1")
    return AgexTraces().read(dump_run(Run(messages=tuple(messages), **kw)))


def user(text, mid="u"):
    return Message(id=mid, role="user", parts=(Text(text=text),))


def reply(*parts, mid="a", model=None, usage=None):
    return Message(id=mid, role="assistant", parts=parts, model=model, usage=usage)


def results(*parts, mid="t"):
    return Message(id=mid, role="tool", parts=parts)


def call(call_id, name="terminal", **args):
    return ToolCall(call_id=call_id, name=name, args=args)


def result(call_id, content, name="terminal", **kw):
    return ToolResult(call_id=call_id, name=name, content=content, **kw)


class TestRun:
    def test_what_the_run_says_of_itself(self):
        run = read(
            user("write a"),
            reply(
                Text(text="Writing."),
                model="claude-sonnet-5-5",
                usage=Usage(input_tokens=100, output_tokens=20, cache_read_tokens=80),
            ),
            run_id="r7",
            status="completed",
            started_at=10.0,
            ended_at=12.5,
            compaction=Usage(input_tokens=5, output_tokens=1, cache_write_tokens=3),
        )
        assert (run.id, run.status, run.started, run.ended) == (
            "r7",
            "completed",
            10.0,
            12.5,
        )
        assert run.model == "claude-sonnet-5-5"
        # the replies' and the summary calls'; agex counts no reasoning
        assert run.tokens == TraceTokens(
            input=105, output=21, cache_read=80, cache_write=3, reasoning=None
        )

    def test_a_run_with_no_status_is_running(self):
        assert read(user("hi")).status == "running"

    def test_a_run_with_no_replies_counts_nothing_and_names_no_model(self):
        run = read(user("hi"))
        assert run.model is None
        assert run.tokens == TraceTokens(0, 0, 0, 0, None)

    def test_each_model_is_named_once_in_the_order_it_replied(self):
        run = read(
            reply(Text(text="a"), mid="a1", model="m-small"),
            reply(Text(text="b"), mid="a2", model="m-large"),
            reply(Text(text="c"), mid="a3", model="m-small"),
        )
        assert run.model == "m-small, m-large"

    def test_agex_stores_no_system_prompt(self):
        assert read(user("hi")).system is None

    def test_a_system_message_the_run_holds_is_its_prompt(self):
        system = Message(id="s", role="system", parts=(Text(text="You write."),))
        run = read(system, user("hi"))
        assert run.system == "You write."
        assert [e.kind for e in run.events] == ["user"]

    @pytest.mark.parametrize(
        "body",
        [
            "a run",
            {},
            {"run_id": 5},
            {"run_id": "r", "messages": [{"id": "m", "role": "bogus"}]},
            {"run_id": "r", "format": 99},
        ],
    )
    def test_a_body_that_is_not_an_agex_run_is_refused(self, body):
        with pytest.raises(ValueError):
            AgexTraces().read(body)


class TestEvents:
    def test_a_reply_reads_in_the_order_it_came(self):
        run = read(
            user("write a"),
            reply(
                Thinking(text="a file named a", signature="sig"),
                Text(text="Writing it."),
                call("c1", "file_write", path="a"),
            ),
            results(result("c1", "wrote a", "file_write")),
            reply(Text(text="Done."), mid="a2"),
        )
        assert run.events == (
            TraceEvent("user", "write a"),
            TraceEvent("thinking", "a file named a"),
            TraceEvent("assistant", "Writing it."),
            TraceEvent("tool", "wrote a", tool="file_write", args={"path": "a"}),
            TraceEvent("assistant", "Done."),
        )

    def test_a_failed_call_is_an_error(self):
        run = read(
            reply(call("c1", command="exit 3")),
            results(result("c1", "exit 3", is_error=True)),
        )
        assert [(e.text, e.error) for e in run.events] == [("exit 3", True)]

    def test_results_are_matched_to_calls_by_id(self):
        run = read(
            reply(call("c1", command="ls"), call("c2", command="pwd")),
            results(result("c2", "/workspace"), result("c1", "a.txt")),
        )
        assert [(e.args, e.text) for e in run.events] == [
            ({"command": "ls"}, "a.txt"),
            ({"command": "pwd"}, "/workspace"),
        ]

    def test_an_id_reused_later_answers_only_the_reply_it_follows(self):
        run = read(
            reply(call("c1", command="ls"), mid="a1"),
            results(result("c1", "first"), mid="t1"),
            reply(call("c1", command="ls"), mid="a2"),
            results(result("c1", "second"), mid="t2"),
        )
        assert [e.text for e in run.events] == ["first", "second"]

    def test_results_spread_over_tool_messages_all_answer_the_reply(self):
        run = read(
            reply(call("c1"), call("c2")),
            results(result("c1", "one"), mid="t1"),
            results(result("c2", "two"), mid="t2"),
        )
        assert [e.text for e in run.events] == ["one", "two"]

    def test_a_call_with_no_result_is_in_flight(self):
        run = read(user("go"), reply(call("c1", command="sleep 9")))
        (tool,) = [e for e in run.events if e.kind == "tool"]
        assert (tool.text, tool.error) == (None, False)

    def test_a_result_no_call_claims_stands_alone(self):
        run = read(
            reply(call("c1")),
            results(result("c1", "mine"), result("c9", "stray", "file_read")),
            results(result("c8", "orphan", "file_write"), mid="t2"),
        )
        assert [(e.tool, e.text, e.args) for e in run.events] == [
            ("terminal", "mine", {}),
            ("file_read", "stray", None),
            ("file_write", "orphan", None),
        ]

    def test_adjacent_text_is_one_event_and_empty_parts_none(self):
        run = read(
            user(""),
            reply(
                Thinking(text=""),
                Text(text="one "),
                Text(text="two"),
                call("c1"),
                Text(text=""),
            ),
            results(result("c1", "")),
        )
        assert run.events == (
            TraceEvent("assistant", "one two"),
            TraceEvent("tool", "", tool="terminal", args={}),
        )

    def test_a_results_images_are_left_out(self):
        run = read(
            reply(call("c1", "view_image", path="p.png")),
            results(
                result(
                    "c1",
                    "an image",
                    "view_image",
                    images=(Image(data="aGk=", media_type="image/png"),),
                )
            ),
        )
        assert [e.text for e in run.events] == ["an image"]


class TestMounted:
    def test_an_agex_session_reads_as_a_trace(self):
        store = Store(memory=True, traces={HARNESS: AgexTraces()})
        worker = store.open("worker")
        reader = store.open("reader")
        try:
            agent = Agent(
                ScriptedProvider(
                    [
                        writes("/workspace/a.txt", "A"),
                        calls("terminal", command="exit 3"),
                        says("wrote a"),
                    ]
                )
            )
            outcome = agent.session(worker).say("write a")
            assert outcome.status == "completed"

            reader.files.attach("worker", "w", trace=True)
            index = reader.files.read("w/.trace/index.md").decode()
            assert "001-" in index and "completed" in index
            (run_dir,) = reader.files.fs.list("/workspace/w/.trace/runs")
            base = f"w/.trace/runs/{run_dir}"
            run_md = reader.files.read(f"{base}/run.md").decode()
            assert "- tool calls: 2 (1 error)" in run_md
            assert "- system prompt: (not kept)" in run_md
            assert "- files changed: 1 (changed.md)" in run_md
            changed = reader.files.read(f"{base}/changed.md").decode()
            assert "- A a.txt" in changed
            events = sorted(reader.files.fs.list(f"/workspace/{base}/events"))
            assert events == [
                "001-user.md",
                "002-tool-file_write.md",
                "003-tool-terminal.md",
                "004-assistant.md",
            ]
            assert (
                "wrote a"
                in reader.files.read(f"{base}/events/004-assistant.md").decode()
            )
            assert not reader.files.fs.exists("/workspace/w/.trace/system")
        finally:
            reader.close()
            worker.close()
            store.close()
