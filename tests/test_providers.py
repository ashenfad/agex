"""The provider protocol, its pydantic-ai implementation and the mapping
between agex's messages and pydantic-ai's, driven by the scripted
provider through pydantic-ai's real request and streaming path."""

import pytest
from nontainer import Store
from nontainer.adapters.tools import Toolset
from nontainer.conformance.corpus import calls, fails, says, thinks, writes
from nontainer.turns import TextDelta, ThinkingDelta
from pydantic_ai import messages as pai
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from agex.providers import Provider, Reply, Settings, ToolSpec
from agex.providers.pydanticai import (
    PydanticAIProvider,
    from_messages,
    from_response,
    to_messages,
)
from agex.providers.scripted import ScriptedProvider
from agex.record import Message, Text, Thinking, ToolCall, ToolResult, Usage

WRITE = ToolSpec(
    name="file_write",
    description="Write a file.",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
)


def user(text):
    return Message(id=f"u-{text}", role="user", parts=(Text(text=text),))


async def collect(provider, messages, tools=(), settings=Settings()):
    events = [e async for e in provider.stream(messages, tools, settings)]
    assert isinstance(events[-1], Reply)
    assert not any(isinstance(e, Reply) for e in events[:-1])
    return events[:-1], events[-1].message


# -- the scripted provider ----------------------------------------------------------


async def test_a_reply_streams_its_text_then_arrives_whole():
    provider = ScriptedProvider([thinks("first, the plan", "Hello there.")])
    assert isinstance(provider, Provider)
    deltas, reply = await collect(provider, [user("hi")])
    assert "".join(d.text for d in deltas if isinstance(d, TextDelta)) == "Hello there."
    assert "".join(d.text for d in deltas if isinstance(d, ThinkingDelta)) == (
        "first, the plan"
    )
    assert reply.role == "assistant" and reply.id
    assert reply.parts == (Thinking(text="first, the plan"), Text(text="Hello there."))
    assert reply.model == "scripted" and reply.usage is not None
    assert reply.usage.input_tokens > 0
    assert provider.name == "function:scripted"


async def test_a_tool_call_round_trips_with_its_provider_id():
    """The call keeps the id the provider gave it, and the next request
    sends the result back under that id, after the call it answers."""
    provider = ScriptedProvider([writes("/workspace/a.txt", "A"), says("Done.")])
    _, reply = await collect(provider, [user("write a")], tools=[WRITE])
    (call,) = [p for p in reply.parts if isinstance(p, ToolCall)]
    assert call == ToolCall(
        call_id="call_1",
        name="file_write",
        args={"path": "/workspace/a.txt", "content": "A"},
    )

    result = Message(
        id="t1",
        role="tool",
        parts=(ToolResult(call_id=call.call_id, name=call.name, content="ok"),),
    )
    _, done = await collect(provider, [user("write a"), reply, result], tools=[WRITE])
    assert done.text == "Done."

    sent = provider.seen[1]
    assert [type(m).__name__ for m in sent] == [
        "ModelRequest",
        "ModelResponse",
        "ModelRequest",
    ]
    (sent_call,) = [p for p in sent[1].parts if isinstance(p, pai.ToolCallPart)]
    (sent_result,) = sent[2].parts
    assert sent_call.tool_call_id == sent_result.tool_call_id == "call_1"
    assert isinstance(sent_result, pai.ToolReturnPart)
    assert sent_result.outcome == "success"


async def test_each_call_gets_an_id_of_its_own():
    provider = ScriptedProvider([calls("t", n=1), calls("t", n=2)])
    _, first = await collect(provider, [user("go")])
    _, second = await collect(provider, [user("go")])
    ids = [
        p.call_id for m in (first, second) for p in m.parts if isinstance(p, ToolCall)
    ]
    assert ids == ["call_1", "call_2"]


async def test_failures_raise_and_a_script_that_runs_out_ends_the_turn():
    provider = ScriptedProvider([fails("provider"), fails("error")])
    with pytest.raises(ModelHTTPError) as info:
        await collect(provider, [user("go")])
    assert info.value.status_code == 529
    with pytest.raises(RuntimeError, match="scripted model failed"):
        await collect(provider, [user("go")])
    _, reply = await collect(provider, [user("go")])
    assert reply.text == "(the script ran out)"


async def test_the_script_can_be_a_callable():
    steps = iter([says("one"), says("two")])
    provider = ScriptedProvider(lambda: next(steps))
    assert (await collect(provider, [user("go")]))[1].text == "one"
    assert (await collect(provider, [user("go")]))[1].text == "two"


# -- what reaches the model -------------------------------------------------------


async def test_tools_and_settings_reach_the_model():
    seen: list[AgentInfo] = []

    async def reply(messages, info):
        seen.append(info)
        yield "ok"

    provider = PydanticAIProvider(FunctionModel(stream_function=reply, model_name="m"))
    settings = Settings(max_tokens=200, temperature=0.5, extra={"top_p": 0.9})
    await collect(provider, [user("go")], tools=[WRITE], settings=settings)
    (info,) = seen
    (tool,) = info.function_tools
    assert (tool.name, tool.description, tool.parameters_json_schema) == (
        WRITE.name,
        WRITE.description,
        WRITE.parameters,
    )
    assert info.model_settings is not None
    keys = ("max_tokens", "temperature", "top_p")
    assert {k: info.model_settings.get(k) for k in keys} == {
        "max_tokens": 200,
        "temperature": 0.5,
        "top_p": 0.9,
    }


def test_thinking_becomes_pydantic_ais_own_setting():
    """pydantic-ai translates it for each provider (and drops it for a
    model that cannot reason, the scripted one included), so what agex
    owes it is the setting itself."""
    from agex.providers.pydanticai import _model_settings

    assert _model_settings(Settings(thinking="high")) == {"thinking": "high"}
    assert _model_settings(Settings(thinking=False)) == {"thinking": False}
    assert _model_settings(Settings()) is None


def test_a_tool_spec_comes_from_a_workspace_tool():
    with Store(memory=True) as store:
        ws = store.open("s")
        (tool,) = [t for t in Toolset(ws).tools() if t.name == "file_write"]
        spec = ToolSpec.of(tool)
        assert (spec.name, spec.description, spec.parameters) == (
            "file_write",
            tool.description,
            dict(tool.parameters),
        )
        ws.close()


# -- the mapping ---------------------------------------------------------------------


def a_conversation() -> list[Message]:
    return [
        Message(id="s", role="system", parts=(Text(text="You write files."),)),
        user("write a"),
        Message(
            id="a1",
            role="assistant",
            parts=(
                Thinking(
                    text="plan",
                    signature="sig-1",
                    provider="anthropic",
                    details={"x": 1},
                ),
                Text(text="Writing."),
                ToolCall(call_id="toolu_1", name="file_write", args={"path": "a"}),
                ToolCall(call_id="toolu_2", name="file_write", args={"path": "b"}),
            ),
            model="claude-sonnet-5-5",
            provider="anthropic",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
                cache_read_tokens=3,
                cache_write_tokens=2,
            ),
        ),
        Message(
            id="t1",
            role="tool",
            parts=(
                ToolResult(call_id="toolu_1", name="file_write", content="ok"),
                ToolResult(
                    call_id="toolu_2",
                    name="file_write",
                    content="disk full",
                    is_error=True,
                ),
            ),
        ),
        Message(
            id="a2", role="assistant", parts=(Text(text="One failed."),), usage=Usage()
        ),
    ]


def _without_ids(messages):
    return [(m.role, m.parts, m.model, m.provider, m.usage) for m in messages]


def test_messages_map_to_pydantic_ai_and_back():
    conversation = a_conversation()
    mapped = to_messages(conversation)
    assert [type(m).__name__ for m in mapped] == [
        "ModelRequest",
        "ModelResponse",
        "ModelRequest",
        "ModelResponse",
    ]
    thinking = mapped[1].parts[0]
    assert isinstance(thinking, pai.ThinkingPart)
    assert (thinking.signature, thinking.provider_name, thinking.provider_details) == (
        "sig-1",
        "anthropic",
        {"x": 1},
    )
    failed = mapped[2].parts[1]
    assert isinstance(failed, pai.ToolReturnPart) and failed.outcome == "failed"
    back = from_messages(mapped)
    assert _without_ids(back) == _without_ids(conversation)
    assert all(m.id for m in back)


def test_a_reply_keeps_cache_usage_and_names_its_model():
    response = pai.ModelResponse(
        parts=[pai.TextPart(content="hi")],
        usage=RequestUsage(
            input_tokens=100,
            output_tokens=7,
            cache_read_tokens=90,
            cache_write_tokens=4,
        ),
        model_name="gpt-5",
        provider_name="openai",
    )
    message = from_response(response, id="m1")
    assert (
        message.id == "m1" and message.model == "gpt-5" and message.provider == "openai"
    )
    assert message.usage == Usage(
        input_tokens=100, output_tokens=7, cache_read_tokens=90, cache_write_tokens=4
    )


def test_a_part_agex_does_not_keep_is_left_out_and_logged(caplog):
    response = pai.ModelResponse(
        parts=[
            pai.TextPart(content="see file"),
            pai.FilePart(content=pai.BinaryContent(data=b"x", media_type="image/png")),
        ]
    )
    message = from_response(response)
    assert message.parts == (Text(text="see file"),)
    assert "left out a file part" in caplog.text


def test_a_requests_parts_keep_their_order():
    """A prompt that followed a tool result in one request still follows
    it: results gather into one tool message only while they run on."""
    request = pai.ModelRequest(
        parts=[
            pai.ToolReturnPart(tool_name="t", content="first", tool_call_id="c1"),
            pai.UserPromptPart(content="also, use tabs"),
            pai.ToolReturnPart(tool_name="t", content="second", tool_call_id="c2"),
            pai.ToolReturnPart(tool_name="t", content="third", tool_call_id="c3"),
        ]
    )
    messages = from_messages([request])
    assert [
        (m.role, [getattr(p, "content", None) or p.text for p in m.parts])
        for m in messages
    ] == [
        ("tool", ["first"]),
        ("user", ["also, use tabs"]),
        ("tool", ["second", "third"]),
    ]
    (back,) = to_messages(messages)
    assert [type(p).__name__ for p in back.parts] == [
        "ToolReturnPart",
        "UserPromptPart",
        "ToolReturnPart",
        "ToolReturnPart",
    ]
    assert [getattr(p, "tool_call_id", None) for p in back.parts] == [
        "c1",
        None,
        "c2",
        "c3",
    ]


def test_which_errors_are_worth_resuming():
    from pydantic_ai.exceptions import ModelAPIError

    provider = ScriptedProvider([])
    for code in (408, 409, 425, 429, 500, 502, 503, 529):
        assert provider.transient(ModelHTTPError(code, "m")), code
    for code in (400, 401, 402, 403, 404, 422):
        assert not provider.transient(ModelHTTPError(code, "m")), code

    class APIConnectionError(Exception):
        """Named as the provider SDKs name it."""

    unreachable = ModelAPIError("m", "Connection error.")
    unreachable.__cause__ = APIConnectionError()
    assert provider.transient(unreachable)
    assert provider.transient(TimeoutError())
    assert provider.transient(ConnectionRefusedError())
    assert not provider.transient(ModelAPIError("m", "the response was filtered"))
    assert not provider.transient(ValueError("a bug"))


def test_a_providers_ids_and_details_go_back_with_its_parts():
    """Gemini signs a tool call (``thought_signature`` in its details),
    OpenAI's Responses API names each reply item (``fc_...``,
    ``rs_...``): what the reply carried is sent back as it came."""
    response = pai.ModelResponse(
        parts=[
            pai.ThinkingPart(
                content="", signature="enc", id="rs_1", provider_name="openai"
            ),
            pai.TextPart(content="ok", id="msg_1", provider_name="openai"),
            pai.ToolCallPart(
                tool_name="file_write",
                args={"path": "a"},
                tool_call_id="call_1",
                id="fc_1",
                provider_name="google",
                provider_details={"thought_signature": "sig=="},
            ),
        ],
        provider_name="openai",
    )
    message = from_response(response, id="m1")
    thinking, text, call = message.parts
    assert (thinking.id, text.id, call.id) == ("rs_1", "msg_1", "fc_1")
    assert call.details == {"thought_signature": "sig=="} and call.provider == "google"
    (back,) = to_messages([message])
    sent_thinking, sent_text, sent_call = back.parts
    assert (sent_thinking.id, sent_thinking.signature) == ("rs_1", "enc")
    assert sent_text.id == "msg_1"
    assert (sent_call.id, sent_call.tool_call_id) == ("fc_1", "call_1")
    assert sent_call.provider_details == {"thought_signature": "sig=="}


def test_an_error_openrouter_relays_from_upstream_is_worth_resuming():
    pytest.importorskip("openai")
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

    provider = PydanticAIProvider(
        OpenRouterModel("meta/x", provider=OpenRouterProvider(api_key="test-key"))
    )
    relayed = {
        "message": "Provider returned error",
        "code": 404,
        "metadata": {"provider_name": "Meta", "provider_error_code": "model_not_found"},
    }
    assert provider.transient(ModelHTTPError(404, "meta/x", body=relayed))
    assert provider.transient(ModelHTTPError(400, "meta/x", body=relayed))
    # from a response or mid-stream, pydantic-ai passes only the message,
    # and the object (if any) rides on the SDK error that caused it
    assert provider.transient(
        ModelHTTPError(404, "meta/x", body="Provider returned error")
    )
    streamed = ModelHTTPError(400, "meta/x", body="the upstream said no")

    class APIError(Exception):
        body = {"error": relayed}

    streamed.__cause__ = APIError()
    assert provider.transient(streamed)
    # OpenRouter's own refusals keep their status's verdict
    assert not provider.transient(
        ModelHTTPError(400, "meta/x", body="meta/x is not a valid model ID")
    )
    own = {"message": "meta/x is not a valid model ID", "code": 400}
    assert not provider.transient(ModelHTTPError(400, "meta/x", body=own))
    assert not provider.transient(ModelHTTPError(402, "meta/x", body={"code": 402}))
    # the rule is OpenRouter's alone
    assert not ScriptedProvider([]).transient(ModelHTTPError(404, "m", body=relayed))
