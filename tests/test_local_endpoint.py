"""agex against an OpenAI-compatible endpoint on this machine, with no
network beyond it: the air-gapped deployment (a local vLLM or Ollama)
served by a fake that streams chat completions the way they do."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

openai = pytest.importorskip("openai")

from nontainer import Store  # noqa: E402
from nontainer.turns import ToolEnded, Usage  # noqa: E402
from pydantic_ai.models.openai import OpenAIChatModel  # noqa: E402
from pydantic_ai.providers.openai import OpenAIProvider  # noqa: E402

from agex import Agent  # noqa: E402


def _chunk(delta=None, finish=None, usage=None):
    body = {
        "id": "chatcmpl-local",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "local",
        "choices": []
        if usage
        else [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = usage
    return body


USAGE = {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}

WRITE = [
    _chunk(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_local_1",
                    "type": "function",
                    "function": {
                        "name": "file_write",
                        "arguments": json.dumps(
                            {"path": "/workspace/a.txt", "content": "A"}
                        ),
                    },
                }
            ],
        }
    ),
    _chunk(finish="tool_calls"),
    _chunk(usage=USAGE),
]

REPLY = [
    _chunk({"role": "assistant", "content": "wrote "}),
    _chunk({"content": "it"}),
    _chunk(finish="stop"),
    _chunk(usage=USAGE),
]


class _Endpoint:
    """A chat-completions endpoint that streams one scripted reply per
    request and keeps the requests it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                endpoint.requests.append(json.loads(self.rfile.read(length)))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for chunk in endpoint.replies.pop(0):
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _model(base_url):
    client = openai.AsyncOpenAI(base_url=base_url, api_key="local", max_retries=0)
    return OpenAIChatModel("local", provider=OpenAIProvider(openai_client=client))


@pytest.fixture
def ws():
    store = Store(memory=True)
    ws = store.open("s")
    yield ws
    ws.close()
    store.close()


def test_a_turn_against_a_local_endpoint(ws):
    endpoint = _Endpoint([WRITE, REPLY])
    try:
        outcome = (
            Agent(_model(endpoint.url), primer="You write files.")
            .session(ws)
            .say("write a")
        )
    finally:
        endpoint.close()
    assert (outcome.status, outcome.text) == ("completed", "wrote it")
    assert ws.files.read("/workspace/a.txt") == b"A"
    (ended,) = [e for e in outcome.events if isinstance(e, ToolEnded)]
    assert ended.call_id == "call_local_1" and not ended.is_error
    assert [e.input_tokens for e in outcome.events if isinstance(e, Usage)] == [50, 50]

    first, second = endpoint.requests
    assert first["stream"] is True
    assert first["messages"][0]["role"] == "system"
    assert first["messages"][0]["content"].startswith("You write files.")
    assert "file_write" in [t["function"]["name"] for t in first["tools"]]
    # the result goes back under the call's own id, after the call
    roles = [m["role"] for m in second["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert second["messages"][2]["tool_calls"][0]["id"] == "call_local_1"
    assert second["messages"][3]["tool_call_id"] == "call_local_1"


def test_an_endpoint_that_cannot_be_reached_interrupts_the_turn(ws):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    # closed now: nothing listens there
    outcome = Agent(_model(f"http://127.0.0.1:{port}/v1")).session(ws).say("hi")
    assert outcome.status == "interrupted"
    assert "Connection" in (outcome.message or "")
