"""Event-log renderer for the provider-native tool-use wire format.

Converts agex :class:`~agex.agent.events.Event` objects into
provider-agnostic message dicts whose ``content`` uses ``tool_use`` and
``tool_result`` blocks.  Clients translate these dicts to the concrete
shape each provider expects.

Rendering rules:

* Each :class:`ActionEvent` emits one assistant message whose content
  is a list of ``tool_use`` / ``text`` / ``thinking`` blocks — one per
  emission, in the order they appear in ``event.emissions``.
* Each tool-call emission (Python / Terminal / FileWrite / FileEdit)
  needs a matching ``tool_result`` in the next user message.
* :class:`~agex.agent.events.OutputEvent` parts carry an
  ``emission_id`` that pairs them back to the emission whose execution
  produced them; the renderer groups observations per emission.
* Terminator events (Success / Fail / Clarify / Cancelled) pair to the
  last actionable emission in the current turn — that's the one whose
  code raised the terminator.
* Non-action events (TaskStart, FileEvent, SystemNoteEvent,
  ChapterEvent) are emitted as text parts in the surrounding user
  message to keep role alternation clean.
"""

from typing import Any, List

from agex.agent.emissions import (
    FileEditEmission,
    FileWriteEmission,
    PythonEmission,
    TerminalEmission,
    TextEmission,
    ThinkingEmission,
)
from agex.agent.events import (
    ActionEvent,
    CancelledEvent,
    ChapterEvent,
    ClarifyEvent,
    ErrorEvent,
    Event,
    FailEvent,
    FileEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.objects import ImageAction, PrintAction
from agex.llm.core import ContentPart, ImagePart, TextPart
from agex.render.primitives import (
    HI_DETAIL_BUDGET,
    render_chapter,
    render_output_parts_full,
    render_task_start,
)
from agex.render.value import render_value

from .schemas import (
    TOOL_EDIT_FILE,
    TOOL_PYTHON,
    TOOL_TERMINAL,
    TOOL_WRITE_FILE,
)


def _content_parts_to_dicts(parts: List[ContentPart]) -> List[dict]:
    out: List[dict] = []
    for p in parts:
        if isinstance(p, TextPart):
            out.append({"type": "text", "text": p.text})
        elif isinstance(p, ImagePart):
            out.append({"type": "image", "image_data": p.image})
    return out


# Defensive per-part cap.  An accidental ``print(huge_base64_blob)``
# would otherwise inflate a single tool_result by a million-plus chars
# and blow the context budget before chaptering ever runs.
_MAX_PART_CHARS = 32768


def _truncate_str(s: str, max_chars: int = _MAX_PART_CHARS) -> str:
    if len(s) <= max_chars:
        return s
    marker = f"\n... [truncated, original was {len(s)} chars]"
    return s[: max_chars - len(marker)] + marker


def _print_action_to_text(action: PrintAction) -> str:
    """Render a :class:`PrintAction` the same way ``print()`` does —
    ``str(arg)`` for each arg joined by spaces, not ``repr``.
    """
    return " ".join(_truncate_str(str(arg)) for arg in action.args)


def _part_to_text_or_image(part: Any) -> tuple[str | None, ImageAction | None]:
    """Classify a single OutputEvent part into text or image.

    Returns ``(text, image)`` where exactly one is non-None.
    """
    if isinstance(part, PrintAction):
        return _print_action_to_text(part), None
    if isinstance(part, ImageAction):
        return None, part
    if isinstance(part, str):
        return _truncate_str(part), None
    return render_value(part, budget=_MAX_PART_CHARS), None


def _tool_use_block_for(emission: Any, block_id: str) -> tuple[dict | None, str | None]:
    """Build a tool_use block for an actionable emission.

    Returns ``(block, tool_name)``.  ``(None, None)`` for
    non-actionable emissions (Text / Thinking).
    """

    def _wrap(name: str, inp: dict) -> dict:
        block: dict[str, Any] = {
            "type": "tool_use",
            "id": block_id,
            "name": name,
            "input": inp,
        }
        if emission.signature is not None:
            block["signature"] = emission.signature
        return block

    if isinstance(emission, PythonEmission):
        inp: dict[str, Any] = {"code": emission.code or ""}
        if emission.title:
            inp["title"] = emission.title
        if emission.thinking:
            inp["thinking"] = emission.thinking
        return _wrap(TOOL_PYTHON, inp), TOOL_PYTHON
    if isinstance(emission, TerminalEmission):
        inp = {"commands": emission.commands or ""}
        if emission.title:
            inp["title"] = emission.title
        if emission.thinking:
            inp["thinking"] = emission.thinking
        return _wrap(TOOL_TERMINAL, inp), TOOL_TERMINAL
    if isinstance(emission, FileWriteEmission):
        inp = {"path": emission.path, "content": emission.content}
        if emission.mode != "write":
            inp["mode"] = emission.mode
        return _wrap(TOOL_WRITE_FILE, inp), TOOL_WRITE_FILE
    if isinstance(emission, FileEditEmission):
        inp = {
            "path": emission.path,
            "search": emission.search,
            "replace": emission.content,
        }
        if emission.match_all:
            inp["match_all"] = True
        return _wrap(TOOL_EDIT_FILE, inp), TOOL_EDIT_FILE
    return None, None


def _synthesize_file_result(emission: Any) -> str:
    """Plain-language tool_result text for file emissions.

    File operations don't produce execution output; the LLM needs a
    string naming what happened so the tool_use → tool_result pairing
    is legible in its in-context reasoning.
    """
    if isinstance(emission, FileWriteEmission):
        verb = "appended to" if emission.mode == "append" else "wrote"
        return f"write_file: {verb} {emission.path}"
    if isinstance(emission, FileEditEmission):
        suffix = " (match_all)" if emission.match_all else ""
        return f"edit_file: replace applied to {emission.path}{suffix}"
    return ""


def _file_event_to_text(event: FileEvent) -> str:
    parts: list[str] = []
    if event.added:
        parts.append(f"Added: {', '.join(event.added)}")
    if event.modified:
        parts.append(f"Modified: {', '.join(event.modified)}")
    if event.removed:
        parts.append(f"Removed: {', '.join(event.removed)}")
    return f"[file changes — {event.file_source}] {'; '.join(parts)}"


def _tool_result_block(
    tool_use_id: str,
    content: str | list[dict],
) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }


def render_events_as_tool_use(events: List[Event]) -> List[dict]:
    """Render events for the tool-use wire format.

    Walks ``event.emissions`` per :class:`ActionEvent` and pairs each
    actionable emission with a ``tool_result`` in the next user
    message.  OutputEvents whose parts carry an ``emission_id`` route
    to that emission's result; parts without an id (or for which no
    matching emission exists in the current turn) fall back to the
    last actionable emission, matching how the old renderer handled
    single-action turns.
    """
    messages: List[dict[str, Any]] = []

    # Per-turn state — reset each time a new ActionEvent arrives.
    tool_use_order: list[tuple[str, str]] = []  # (emission_id, tool_name) in order
    obs_by_emission: dict[str, tuple[list[str], list[ImageAction]]] = {}
    synth_by_emission: dict[str, str] = {}
    terminator_emission_id: str | None = None
    terminator_text: str | None = None

    # Trailing text parts that belong to the *next* user message
    # (TaskStart banners, FileEvents, ChapterEvents, SystemNoteEvents).
    # Kept separate so they land *after* tool_result blocks in the
    # emitted user message — Anthropic wants tool_results at the top.
    pending_text: list[dict] = []

    task_number = 0
    # Walk with the *raw* event-log index so block_ids match the loop's
    # stamp on PrintAction / ImageAction.  ErrorEvents are skipped in
    # rendering but still consume an index.
    indexed_events = [
        (i, e) for i, e in enumerate(events) if not isinstance(e, ErrorEvent)
    ]

    def _last_actionable() -> str | None:
        if not tool_use_order:
            return None
        return tool_use_order[-1][0]

    def _reset_turn() -> None:
        nonlocal terminator_emission_id, terminator_text
        tool_use_order.clear()
        obs_by_emission.clear()
        synth_by_emission.clear()
        terminator_emission_id = None
        terminator_text = None

    def _build_tool_result(emission_id: str, tool_name: str) -> dict:
        synth = synth_by_emission.get(emission_id)
        text_bits, image_parts = obs_by_emission.get(emission_id, ([], []))
        is_terminator = emission_id == terminator_emission_id
        term_text = terminator_text if is_terminator else None

        # File emissions record a synthesized "wrote X" / "edit applied"
        # success line because the VFS apply itself produces no output
        # on the happy path.  But a FAILED apply (e.g. edit_file with a
        # non-matching search) logs its error via create_error_output,
        # which lands in ``obs_by_emission[emission_id]``.  Using the
        # synth blindly when observations exist would tell the agent
        # the file was written even though it wasn't — so the synth
        # only stands in when there are genuinely no observations.
        if synth is not None and not text_bits and not image_parts:
            content = synth if not term_text else f"{synth}\n{term_text}"
            return _tool_result_block(emission_id, content)

        text_body = "\n".join(b for b in text_bits if b)

        if image_parts:
            blocks: list[dict] = []
            header = f"{tool_name}: output" if text_body else f"{tool_name}:"
            if text_body:
                blocks.append({"type": "text", "text": f"{header}\n{text_body}"})
            else:
                blocks.append({"type": "text", "text": header})
            rendered, _ = render_output_parts_full(image_parts, budget=HI_DETAIL_BUDGET)
            for p in rendered:
                if isinstance(p, ImagePart):
                    blocks.append({"type": "image", "image_data": p.image})
            if term_text:
                blocks.append({"type": "text", "text": term_text})
            return _tool_result_block(emission_id, blocks)

        if text_body and term_text:
            content = f"{tool_name}: output\n{text_body}\n{term_text}"
        elif text_body:
            content = f"{tool_name}: output\n{text_body}"
        elif term_text:
            content = f"{tool_name}: {term_text}"
        else:
            content = f"{tool_name}: (no observation)"
        return _tool_result_block(emission_id, content)

    def _flush_user() -> None:
        parts: list[dict] = []
        for emission_id, tool_name in tool_use_order:
            parts.append(_build_tool_result(emission_id, tool_name))
        parts.extend(pending_text)
        if parts:
            messages.append({"role": "user", "content": parts})
        _reset_turn()
        pending_text.clear()

    def _route_part(part: Any) -> None:
        emission_id: str | None = None
        if isinstance(part, (PrintAction, ImageAction)):
            emission_id = part.emission_id
        if emission_id is None or emission_id not in dict(tool_use_order):
            # Fallback to the last actionable emission — covers raw
            # string parts and legacy events whose parts weren't
            # stamped with an emission_id.
            emission_id = _last_actionable()
        if emission_id is None:
            return  # Stray observation before any action; drop.
        text, image = _part_to_text_or_image(part)
        slot = obs_by_emission.setdefault(emission_id, ([], []))
        if image is not None:
            slot[1].append(image)
        elif text:
            slot[0].append(text)

    for idx, event in indexed_events:
        if isinstance(event, TaskStartEvent):
            task_number += 1
            text, _ = render_task_start(event.message, budget=HI_DETAIL_BUDGET)
            pending_text.append({"type": "text", "text": f"[{task_number}] {text}"})

        elif isinstance(event, ActionEvent):
            # Close out the previous turn.
            _flush_user()

            assistant_content: list[dict] = []
            for j, emission in enumerate(event.emissions):
                if isinstance(emission, TextEmission):
                    if emission.text:
                        assistant_content.append(
                            {"type": "text", "text": emission.text}
                        )
                elif isinstance(emission, ThinkingEmission):
                    # Native-thinking providers (Gemini 3, Claude 4.6+)
                    # expect signed thought parts to round-trip at
                    # their original position.  Emit a ``thinking``
                    # block so the provider-specific translator can
                    # reconstruct the right shape.  Translators that
                    # don't understand ``thinking`` blocks can fall
                    # back to ignoring them or rendering as text.
                    if emission.signature is not None or emission.redacted:
                        block: dict[str, Any] = {"type": "thinking"}
                        if emission.text:
                            block["text"] = emission.text
                        if emission.signature is not None:
                            block["signature"] = emission.signature
                        if emission.redacted:
                            block["redacted"] = True
                        assistant_content.append(block)
                    elif emission.text:
                        # Unsigned narration — fall back to a plain
                        # text block with a visible tag.
                        assistant_content.append(
                            {"type": "text", "text": f"[thinking] {emission.text}"}
                        )
                else:
                    block_id = f"em_{idx}_{j}"
                    block, tool_name = _tool_use_block_for(emission, block_id)
                    if block is not None and tool_name is not None:
                        assistant_content.append(block)
                        tool_use_order.append((block_id, tool_name))
                        if isinstance(emission, (FileWriteEmission, FileEditEmission)):
                            synth_by_emission[block_id] = _synthesize_file_result(
                                emission
                            )

            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})

        elif isinstance(event, OutputEvent):
            for part in event.parts:
                _route_part(part)

        elif isinstance(event, SuccessEvent):
            if tool_use_order:
                terminator_emission_id = _last_actionable()
                estimated = HI_DETAIL_BUDGET * 4
                rendered = render_value(
                    event.result,
                    budget=estimated,
                    token_budget=HI_DETAIL_BUDGET,
                )
                terminator_text = f"task_success returned\n{rendered}"

        elif isinstance(event, CancelledEvent):
            if tool_use_order:
                terminator_emission_id = _last_actionable()
                terminator_text = (
                    f"cancelled after {event.iterations_completed} iterations"
                )

        elif isinstance(event, FailEvent):
            if tool_use_order:
                terminator_emission_id = _last_actionable()
                terminator_text = f"task_fail: {event.message}"

        elif isinstance(event, ClarifyEvent):
            if tool_use_order:
                terminator_emission_id = _last_actionable()
                terminator_text = f"task_clarify: {event.message}"

        elif isinstance(event, FileEvent):
            pending_text.append({"type": "text", "text": _file_event_to_text(event)})

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            pending_text.append({"type": "text", "text": text})

        elif isinstance(event, SystemNoteEvent):
            pending_text.append({"type": "text", "text": event.message})

    _flush_user()
    return messages
