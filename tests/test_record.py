"""The stored run record: agex's own format, read and written as JSON."""

import json

import pytest
from nontainer.conformance.codec import json_schema

from agex.record import (
    FORMAT,
    Message,
    Run,
    Text,
    Thinking,
    ToolCall,
    ToolResult,
    Usage,
    dump_run,
    load_run,
    new_id,
)


def a_run() -> Run:
    return Run(
        run_id="r1",
        status="completed",
        started_at=1.0,
        ended_at=2.5,
        messages=(
            Message(id="m1", role="system", parts=(Text(text="You write files."),)),
            Message(id="m2", role="user", parts=(Text(text="write a"),)),
            Message(
                id="m3",
                role="assistant",
                parts=(
                    Thinking(
                        text="a file named a",
                        signature="sig",
                        provider="anthropic",
                        details={"redacted": False},
                    ),
                    Text(text="Writing it."),
                    ToolCall(call_id="toolu_1", name="file_write", args={"path": "a"}),
                ),
                model="claude-sonnet-5-5",
                provider="anthropic",
                usage=Usage(input_tokens=100, output_tokens=20, cache_read_tokens=80),
            ),
            Message(
                id="m4",
                role="tool",
                parts=(ToolResult(call_id="toolu_1", name="file_write", content="ok"),),
            ),
            Message(
                id="m5",
                role="assistant",
                parts=(Text(text="Done."),),
                usage=Usage(input_tokens=130, output_tokens=5, cache_write_tokens=30),
            ),
        ),
    )


def test_a_run_reads_back_from_its_json():
    run = a_run()
    data = json.loads(json.dumps(dump_run(run)))
    assert data["format"] == FORMAT
    assert data["messages"][2]["parts"][2] == {
        "kind": "tool_call",
        "call_id": "toolu_1",
        "name": "file_write",
        "args": {"path": "a"},
    }
    assert load_run(data) == run


def test_the_record_fits_its_json_schema():
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(dump_run(a_run()), json_schema(Run, title="Run"))


def test_a_run_sums_its_usage_and_a_message_joins_its_text():
    run = a_run()
    assert run.usage == Usage(
        input_tokens=230, output_tokens=25, cache_read_tokens=80, cache_write_tokens=30
    )
    assert run.messages[2].text == "Writing it."
    assert Run(run_id="r2").usage == Usage()


def test_the_loader_refuses_what_it_cannot_read():
    data = dump_run(a_run())
    with pytest.raises(ValueError, match="newer"):
        load_run({**data, "format": FORMAT + 1})
    bad = json.loads(json.dumps(data))
    bad["messages"][0]["parts"][0]["kind"] = "hologram"
    with pytest.raises(ValueError, match="hologram"):
        load_run(bad)
    bad = json.loads(json.dumps(data))
    bad["status"] = "sideways"
    with pytest.raises(ValueError, match="sideways"):
        load_run(bad)


def test_ids_are_fresh():
    assert len({new_id() for _ in range(100)}) == 100
