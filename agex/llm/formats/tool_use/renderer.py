"""Event-log renderer for the provider-native tool-use wire format.

Converts agex :class:`~agex.agent.events.Event` objects into
provider-agnostic message dicts whose ``content`` uses ``tool_use`` and
``tool_result`` blocks.  Clients translate these dicts to the concrete
shape each provider expects (Anthropic content arrays, OpenAI
``tool_calls`` + ``tool`` role messages, etc.).

Rendering rules:

* Each :class:`ActionEvent` emits one assistant message whose content is
  a list of ``tool_use`` blocks — one per :class:`FileAction` /
  :class:`EditAction` (in order) followed by one for the main
  python_action / terminal_action.
* Each ``tool_use`` needs a matching ``tool_result`` in the next user
  message.  File actions get a synthesized result that names what ran
  ("write_file: wrote /helpers/x.py") so the LLM can tie the
  ``tool_result`` back to its earlier ``tool_use`` in plain language;
  the main action's result is filled by the next
  :class:`OutputEvent` / :class:`SuccessEvent` / :class:`CancelledEvent`
  / :class:`FailEvent` / :class:`ClarifyEvent` with a tool-name prefix
  so the same linkage is legible.
* Non-action events (:class:`TaskStartEvent`, :class:`FileEvent`,
  :class:`SystemNoteEvent`, :class:`ChapterEvent`) are emitted as text
  parts within the surrounding user message to keep role alternation
  clean.
"""

from typing import Any, List

from agex.agent.datatypes import EditAction, FileAction
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


# Defensive per-part cap.  An accidental ``print(huge_base64_blob)`` or
# ``task_continue({"data": <megabyte_string>})`` would otherwise inflate
# a single tool_result by a million-plus chars and blow the context
# budget before chaptering ever runs.  32k chars ≈ 8k tokens — generous
# enough that legitimate large outputs survive in full but small enough
# that runaway values get clipped with a marker the LLM can read.
_MAX_PART_CHARS = 32768


def _truncate_str(s: str, max_chars: int = _MAX_PART_CHARS) -> str:
    if len(s) <= max_chars:
        return s
    marker = f"\n... [truncated, original was {len(s)} chars]"
    return s[: max_chars - len(marker)] + marker


def _print_action_to_text(action: PrintAction) -> str:
    """Render a ``PrintAction`` (tuple of ``print()`` args) the same
    way a real ``print()`` does — ``str(arg)`` for each, joined by
    spaces.

    Why ``str()`` instead of ``render_value``:

    - ``render_value`` wraps strings in ``repr``-style quotes, so
      ``print("hello")`` would otherwise show as ``'hello'``.
    - The studio UI uses unbudgeted ``str(item)`` so the LLM-facing
      view should match.

    Each arg's str is bounded by ``_MAX_PART_CHARS`` as a defensive
    cap against runaway values (huge base64 blobs, etc.).
    """
    return " ".join(_truncate_str(str(arg)) for arg in action)


def _output_to_text(event: OutputEvent) -> tuple[str, list[ContentPart]]:
    """Split an OutputEvent into its text stream and image parts.

    The text stream is built verbatim from :class:`PrintAction`\\ s so
    ``print("hello")`` appears as ``hello`` — not ``'hello'``.  Images
    are rendered via the budget-aware path.  Non-print, non-image,
    non-string parts (e.g. dicts handed to ``task_continue``) go
    through ``render_value`` so we get reprobate's structure-aware
    truncation rather than ``str()``-ing a megabyte blob whole.
    """
    text_bits: list[str] = []
    image_parts: list[Any] = []
    for item in event.parts:
        if isinstance(item, PrintAction):
            text_bits.append(_print_action_to_text(item))
        elif isinstance(item, ImageAction):
            image_parts.append(item)
        elif isinstance(item, str):
            text_bits.append(_truncate_str(item))
        else:
            text_bits.append(render_value(item, budget=_MAX_PART_CHARS))
    # Route images through the existing budget-aware renderer so we
    # reuse its PNG-serialization / detail-level logic.
    rendered_images: list[ContentPart] = []
    if image_parts:
        rendered, _ = render_output_parts_full(image_parts, budget=HI_DETAIL_BUDGET)
        rendered_images = [p for p in rendered if isinstance(p, ImagePart)]
    return "\n".join(text_bits), rendered_images


def _build_action_blocks(
    event: ActionEvent, task_number: int, event_index: int
) -> tuple[list[dict], list[tuple[str, str, str]], tuple[str, str]]:
    """Return ``(tool_use_blocks, file_infos, main_info)``.

    - ``file_infos`` is a list of ``(block_id, tool_name, path)`` — one
      per emitted file tool_use, used to synthesize richer
      ``tool_result`` text than a bare ``"ok"``.
    - ``main_info`` is ``(block_id, tool_name)`` for the main action,
      used to prefix paired observations with ``"{tool_name}: ..."``.
    """
    blocks: list[dict] = []
    file_infos: list[tuple[str, str, str]] = []

    for j, fa in enumerate(event.file_actions):
        block_id = f"toolu_{task_number}_{event_index}_{j}"
        if isinstance(fa, FileAction):
            file_input: dict[str, Any] = {
                "path": fa.path,
                "content": fa.content,
            }
            if fa.mode != "write":
                file_input["mode"] = fa.mode
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": TOOL_WRITE_FILE,
                    "input": file_input,
                }
            )
            file_infos.append((block_id, TOOL_WRITE_FILE, fa.path))
        elif isinstance(fa, EditAction):
            edit_input: dict[str, Any] = {"path": fa.path, "search": fa.search}
            if fa.operation == "insert-after":
                edit_input["insert_after"] = fa.content
            elif fa.operation == "insert-before":
                edit_input["insert_before"] = fa.content
            else:
                edit_input["replace"] = fa.content
            if fa.match_all:
                edit_input["match_all"] = True
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block_id,
                    "name": TOOL_EDIT_FILE,
                    "input": edit_input,
                }
            )
            file_infos.append((block_id, TOOL_EDIT_FILE, fa.path))

    main_id = f"toolu_{task_number}_{event_index}_main"
    if event.terminal:
        main_input: dict[str, Any] = {
            "title": event.title,
            "thinking": event.thinking,
            "commands": event.terminal,
        }
        if getattr(event, "report", ""):
            main_input["report"] = event.report
        blocks.append(
            {
                "type": "tool_use",
                "id": main_id,
                "name": TOOL_TERMINAL,
                "input": main_input,
            }
        )
        main_tool_name = TOOL_TERMINAL
    else:
        main_input = {
            "title": event.title,
            "thinking": event.thinking,
            "code": event.code or "",
        }
        if getattr(event, "report", ""):
            main_input["report"] = event.report
        blocks.append(
            {
                "type": "tool_use",
                "id": main_id,
                "name": TOOL_PYTHON,
                "input": main_input,
            }
        )
        main_tool_name = TOOL_PYTHON

    return blocks, file_infos, (main_id, main_tool_name)


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


def _file_action_result_text(action_block: dict) -> str:
    """Synthesize a tool_result that names the tool and what it touched.

    Gives the LLM a plain-language link between ``tool_use`` and
    ``tool_result`` — structural id matching isn't always enough to make
    the continuity legible in the model's in-context reasoning.
    """
    name = action_block["name"]
    inp = action_block.get("input", {})
    path = inp.get("path", "")
    if name == TOOL_WRITE_FILE:
        mode = inp.get("mode", "write")
        verb = "appended to" if mode == "append" else "wrote"
        return f"write_file: {verb} {path}"
    if name == TOOL_EDIT_FILE:
        if "replace" in inp:
            op = "replace"
        elif "insert_after" in inp:
            op = "insert-after"
        elif "insert_before" in inp:
            op = "insert-before"
        else:
            op = "edit"
        suffix = " (match_all)" if inp.get("match_all") else ""
        return f"edit_file: {op} applied to {path}{suffix}"
    return f"{name}: applied"


def render_events_as_tool_use(events: List[Event]) -> List[dict]:
    """Render events for the tool-use wire format.

    Per turn, everything between one :class:`ActionEvent` and the next
    aggregates into a **single** ``tool_result``.  A Python turn can
    produce multiple ``OutputEvent``\\ s — one per ``print()`` call,
    one per ``view_image()``, etc. (see
    :func:`agex.eval.bridge.result.handle_result`) — and those may be
    followed by a ``SuccessEvent`` / ``FailEvent`` / ``ClarifyEvent`` /
    ``CancelledEvent``.  An earlier version of this renderer paired
    only the first such event with the ``tool_result`` and silently
    dropped the rest, so the LLM saw only the first ``print`` of a
    multi-print turn.  This version accumulates all of them.
    """
    messages: List[dict[str, Any]] = []

    # Per-action accumulators.  Reset each time a new ActionEvent
    # begins.  file_tool_results comes from the action's own
    # FileAction/EditAction list; main_* accumulate everything between
    # this action and the next.
    file_tool_results: list[dict] = []
    main_info: tuple[str, str] | None = None  # (block_id, tool_name)
    main_text_bits: list[str] = []
    main_image_parts: list[Any] = []
    main_terminator: str | None = None  # task_success/task_fail/cancelled

    # Trailing text content that belongs to the following user message
    # (task starts, chapter / system note banners, FileEvents).  Kept
    # separate so it lands *after* the tool_result blocks in the
    # emitted user message — Anthropic wants tool_results at the top.
    pending_text: list[dict] = []

    task_number = 0
    filtered = [e for e in events if not isinstance(e, ErrorEvent)]

    def _build_main_tool_result() -> dict | None:
        if main_info is None:
            return None
        main_id, main_name = main_info
        text_body = "\n".join(b for b in main_text_bits if b)

        if main_image_parts:
            blocks: list[dict] = []
            header = f"{main_name}: output" if text_body else f"{main_name}:"
            if text_body:
                blocks.append({"type": "text", "text": f"{header}\n{text_body}"})
            else:
                blocks.append({"type": "text", "text": header})
            for img in main_image_parts:
                blocks.append({"type": "image", "image_data": img.image})
            if main_terminator:
                blocks.append({"type": "text", "text": main_terminator})
            return _tool_result_block(main_id, blocks)

        if text_body and main_terminator:
            content = f"{main_name}: output\n{text_body}\n{main_terminator}"
        elif text_body:
            content = f"{main_name}: output\n{text_body}"
        elif main_terminator:
            content = f"{main_name}: {main_terminator}"
        else:
            content = f"{main_name}: (no observation)"
        return _tool_result_block(main_id, content)

    def flush_user() -> None:
        nonlocal main_info, main_text_bits, main_image_parts, main_terminator
        nonlocal file_tool_results, pending_text
        parts: list[dict] = []
        parts.extend(file_tool_results)
        main_block = _build_main_tool_result()
        if main_block is not None:
            parts.append(main_block)
        parts.extend(pending_text)
        if parts:
            messages.append({"role": "user", "content": parts})
        # Reset all per-turn accumulators.
        file_tool_results = []
        main_info = None
        main_text_bits = []
        main_image_parts = []
        main_terminator = None
        pending_text = []

    for idx, event in enumerate(filtered):
        if isinstance(event, TaskStartEvent):
            task_number += 1
            text, _ = render_task_start(event.message, budget=HI_DETAIL_BUDGET)
            pending_text.append({"type": "text", "text": f"[{task_number}] {text}"})

        elif isinstance(event, ActionEvent):
            # Close out the previous turn's tool_results and text.
            flush_user()
            blocks, file_infos, info = _build_action_blocks(event, task_number, idx)
            messages.append({"role": "assistant", "content": blocks})
            # Synthesize per-file tool_results with tool-name framing so
            # the LLM sees a legible linkage back to each file tool_use.
            for (file_id, _tool, _path), block in zip(
                file_infos, blocks[: len(file_infos)]
            ):
                file_tool_results.append(
                    _tool_result_block(file_id, _file_action_result_text(block))
                )
            main_info = info

        elif isinstance(event, OutputEvent):
            if main_info is None:
                continue  # Stray observation before any action — drop.
            text, image_parts = _output_to_text(event)
            if text:
                main_text_bits.append(text)
            main_image_parts.extend(image_parts)

        elif isinstance(event, SuccessEvent):
            if main_info is not None:
                estimated = HI_DETAIL_BUDGET * 4
                rendered = render_value(
                    event.result,
                    budget=estimated,
                    token_budget=HI_DETAIL_BUDGET,
                )
                main_terminator = f"task_success returned\n{rendered}"

        elif isinstance(event, CancelledEvent):
            if main_info is not None:
                main_terminator = (
                    f"cancelled after {event.iterations_completed} iterations"
                )

        elif isinstance(event, FailEvent):
            if main_info is not None:
                main_terminator = f"task_fail: {event.message}"

        elif isinstance(event, ClarifyEvent):
            if main_info is not None:
                main_terminator = f"task_clarify: {event.message}"

        elif isinstance(event, FileEvent):
            pending_text.append({"type": "text", "text": _file_event_to_text(event)})

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            pending_text.append({"type": "text", "text": text})

        elif isinstance(event, SystemNoteEvent):
            pending_text.append({"type": "text", "text": event.message})

    flush_user()
    return messages
