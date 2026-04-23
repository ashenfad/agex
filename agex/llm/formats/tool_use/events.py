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


@dataclass(frozen=True, slots=True)
class TextPart:
    """A user-facing text ``Part`` emitted alongside (or instead of)
    tool calls.

    Gemini 3 sometimes returns a plain text reply when it could have
    used a tool — e.g., after a confusing tool_result it may switch
    to narrating what went wrong instead of recovering via a new
    tool call.  Dropping that text silently leaves the turn looking
    empty and gives the model nothing to see on replay, which
    encourages another round of the same.  Capturing it as a
    :class:`~agex.agent.emissions.TextEmission` keeps the model's
    output visible in the event log and round-trips it on the next
    request.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ThinkingPart:
    """A native-thinking ``Part`` delivered alongside tool calls.

    Gemini 3 emits thought parts that carry a ``thought_signature``
    even when no function_call accompanies the Part.  Gemini's rules
    require such signatures to be replayed *at the same position* on
    subsequent turns (docs: "if it was returned in a thought part, it
    must be returned in a thought part").  The parser materializes
    these as :class:`~agex.agent.emissions.ThinkingEmission`\\ s so
    they ride through the emission list and the renderer can put them
    back as thought parts when building the next request.
    """

    signature: bytes | None = None
    text: str | None = None
    redacted: bool = False


ToolCallEvent = Union[
    ToolCallStart, ToolCallArgDelta, ToolCallEnd, TextPart, ThinkingPart
]
