"""Emission types — the ordered units the LLM produces in one assistant turn.

An :class:`~agex.agent.events.ActionEvent` is a list of these, in the order
the model emitted them. Each type maps cleanly to a provider-native block:

- :class:`TextEmission` → Anthropic text blocks / Gemini text parts /
  OpenAI assistant ``content``.
- :class:`ThinkingEmission` → provider thinking blocks (signature-bearing)
  for native-thinking models; narration fallback for non-native providers.
- :class:`PythonEmission` / :class:`TerminalEmission` /
  :class:`FileWriteEmission` / :class:`FileEditEmission` → tool_use /
  tool_calls / function_call blocks depending on provider.

The optional ``signature`` field on each emission carries any opaque
per-block state the provider requires to round-trip across turns
(Gemini's ``thought_signature``, Claude's thinking-block signatures).
"""

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class TextEmission:
    """User-facing prose from the model.

    Replaces the old ``report`` field. In provider-native tool use this is
    just a text block the model emits alongside tool calls — no schema
    parameter, no primer instruction. Multiple per turn is idiomatic.
    """

    text: str
    signature: bytes | None = None


@dataclass
class ThinkingEmission:
    """Agent's internal reasoning trace.

    For native-thinking providers (Claude extended thinking, Gemini
    thinking mode), this carries the provider's thinking block content
    plus the signature required to round-trip it across turns.
    ``redacted=True`` marks blocks the provider hid from us but still
    requires us to replay (Claude's ``redacted_thinking``).

    For non-native providers, this is the content of the agex
    narration-in-schema fallback (a ``thinking`` parameter on the
    tool call).
    """

    text: str
    signature: bytes | None = None
    redacted: bool = False


@dataclass
class PythonEmission:
    """A Python code block for the sandbox to execute.

    Maps to the ``python_action`` tool call. In a multi-emission turn,
    PythonEmissions execute sequentially with a shared namespace (as
    if their code bodies were concatenated).

    ``thinking`` carries narration-via-schema reasoning when the
    provider doesn't emit native thinking blocks.  It rides on the
    action emission for round-trip fidelity: a replayed turn shows the
    same ``thinking`` argument the model originally produced.  Phase 4
    drops this field and switches to a separate
    :class:`ThinkingEmission` for native-thinking providers.
    """

    code: str
    title: str | None = None
    thinking: str | None = None
    signature: bytes | None = None


@dataclass
class TerminalEmission:
    """A shell command block for the setup-phase terminal.

    Maps to the ``terminal_action`` tool call. Each TerminalEmission
    executes as one shell invocation; multiple in a turn run in
    sequence.

    ``thinking`` mirrors :class:`PythonEmission.thinking` — narration
    lives on the action emission for round-trip fidelity.
    """

    commands: str
    title: str | None = None
    thinking: str | None = None
    signature: bytes | None = None


@dataclass
class FileWriteEmission:
    """A file write (create or append) requested by the agent.

    Maps to the ``write_file`` tool call.
    """

    path: str
    content: str
    mode: Literal["write", "append"] = "write"
    title: str | None = None
    signature: bytes | None = None


@dataclass
class FileEditEmission:
    """A file edit (search + replace) requested by the agent.

    Maps to the ``edit_file`` tool call: swap ``search`` for ``content``.
    Inserting relative to an anchor is expressed as a replace whose
    ``content`` includes the anchor — e.g. to append ``new_fn`` after
    ``old_fn``, search for ``old_fn`` and replace with
    ``old_fn\\nnew_fn``.
    """

    path: str
    search: str
    content: str
    match_all: bool = False
    title: str | None = None
    signature: bytes | None = None


Emission = Union[
    TextEmission,
    ThinkingEmission,
    PythonEmission,
    TerminalEmission,
    FileWriteEmission,
    FileEditEmission,
]


ACTION_EMISSION_TYPES = (
    PythonEmission,
    TerminalEmission,
    FileWriteEmission,
    FileEditEmission,
)
"""Emission types that represent actionable tool calls (things the
execution loop runs). Text and Thinking emissions are logged but not
executed."""
