"""Provider-agnostic tool-call stream events.

Each LLM client translates its provider's native streaming events
(Anthropic ``content_block_start`` / ``input_json_delta`` /
``content_block_stop``; OpenAI ``tool_calls[].function.arguments``
deltas; Gemini function-call parts) into this small vocabulary so the
tool-use wire format can parse tool calls without knowing the provider.

A single tool call moves through:

    ToolCallStart(id, tool_name)
      -> ToolCallArgDelta(id, argument_chunk)   [0..N, partial JSON]
      -> ToolCallEnd(id)

Multiple tool calls can interleave in the stream as long as each one's
events share its ``call_id``.
"""

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """A tool call is beginning. Carries the tool name.

    ``signature`` carries any opaque per-call state the provider wants
    us to round-trip on subsequent turns (Gemini's ``thought_signature``
    is the load-bearing case today).  ``None`` when the provider
    doesn't sign function calls.
    """

    call_id: str
    tool_name: str
    signature: bytes | None = None


@dataclass(frozen=True, slots=True)
class ToolCallArgDelta:
    """Partial JSON fragment for a tool call's ``input`` arguments.

    Fragments for a given ``call_id`` concatenate to form a valid JSON
    object once the call ends. They may split at any byte boundary,
    including mid-escape and mid-``\\uXXXX``.
    """

    call_id: str
    argument_chunk: str


@dataclass(frozen=True, slots=True)
class ToolCallEnd:
    """A tool call has finished. No more ``ArgDelta``\\ s for this id."""

    call_id: str


ToolCallEvent = Union[ToolCallStart, ToolCallArgDelta, ToolCallEnd]
