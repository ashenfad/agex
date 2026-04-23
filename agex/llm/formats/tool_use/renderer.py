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


def _print_action_to_text(action: PrintAction) -> str:
    """Render a ``PrintAction`` (tuple of ``print()`` args) the same
    way a real ``print()`` does — ``str(arg)`` for each, joined by
    spaces.

    Avoids two prior bugs at once:

    - ``render_value`` wraps strings in ``repr``-style quotes, so
      ``print("hello")`` would otherwise show as ``'hello'`` in the
      tool_result text.
    - ``render_value`` defaults to a 2048-char budget per arg, which
      silently truncated the LLM's view of large printed values
      (e.g. a ``print(test_app(...))`` returning a long result list)
      while the human-facing activity log used unbudgeted ``str(item)``
      and showed the whole thing.

    ``str()`` matches what the studio UI does and what Python's print
    semantically does.  Aggregate token-budget protection is the
    chaptering layer's job.
    """
    return " ".join(str(arg) for arg in action)


def _output_to_text(event: OutputEvent) -> tuple[str, list[ContentPart]]:
    """Split an OutputEvent into its text stream and image parts.

    The text stream is built verbatim from :class:`PrintAction`\\ s so
    ``print("hello")`` appears as ``hello`` — not ``'hello'``.  Images
    are rendered via the budget-aware path.
    """
    text_bits: list[str] = []
    image_parts: list[Any] = []
    for item in event.parts:
        if isinstance(item, PrintAction):
            text_bits.append(_print_action_to_text(item))
        elif isinstance(item, ImageAction):
            image_parts.append(item)
        elif isinstance(item, str):
            text_bits.append(item)
        else:
            # Unknown part type — convert via ``str()`` so the LLM sees
            # whatever the studio UI shows (which also uses ``str()``).
            # Avoids the budget-driven truncation ``render_value`` would
            # impose silently.
            text_bits.append(str(item))
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
    """Render events for the tool-use wire format."""
    messages: List[dict[str, Any]] = []
    pending_user: list[dict] = []
    # Tool-use id + name of the most recent ActionEvent still awaiting
    # its main tool_result.  Cleared once paired.
    pending_main: tuple[str, str] | None = None

    task_number = 0
    filtered = [e for e in events if not isinstance(e, ErrorEvent)]

    def flush_user() -> None:
        nonlocal pending_user, pending_main
        if pending_main is not None:
            main_id, main_name = pending_main
            pending_user.append(
                _tool_result_block(main_id, f"{main_name}: (no observation)")
            )
            pending_main = None
        if pending_user:
            messages.append({"role": "user", "content": pending_user})
            pending_user = []

    for idx, event in enumerate(filtered):
        if isinstance(event, TaskStartEvent):
            task_number += 1
            text, _ = render_task_start(event.message, budget=HI_DETAIL_BUDGET)
            pending_user.append({"type": "text", "text": f"[{task_number}] {text}"})

        elif isinstance(event, ActionEvent):
            flush_user()
            blocks, file_infos, main_info = _build_action_blocks(
                event, task_number, idx
            )
            messages.append({"role": "assistant", "content": blocks})
            # Synthesize per-file tool_results with tool-name framing so
            # the LLM sees a legible linkage back to each file tool_use.
            for (file_id, _tool, _path), block in zip(
                file_infos, blocks[: len(file_infos)]
            ):
                pending_user.append(
                    _tool_result_block(file_id, _file_action_result_text(block))
                )
            pending_main = main_info

        elif isinstance(event, OutputEvent):
            if pending_main is None:
                continue  # Stray observation before any action — drop.
            main_id, main_name = pending_main
            text, image_parts = _output_to_text(event)
            if image_parts:
                blocks: list[dict] = []
                prefix = f"{main_name}: output"
                if text:
                    blocks.append({"type": "text", "text": f"{prefix}\n{text}"})
                else:
                    blocks.append({"type": "text", "text": prefix})
                for img in image_parts:
                    blocks.append({"type": "image", "image_data": img.image})
                pending_user.append(_tool_result_block(main_id, blocks))
            elif text:
                pending_user.append(
                    _tool_result_block(main_id, f"{main_name}: output\n{text}")
                )
            else:
                pending_user.append(
                    _tool_result_block(main_id, f"{main_name}: (no output)")
                )
            pending_main = None

        elif isinstance(event, SuccessEvent):
            if pending_main is not None:
                main_id, main_name = pending_main
                estimated = HI_DETAIL_BUDGET * 4
                rendered = render_value(
                    event.result,
                    budget=estimated,
                    token_budget=HI_DETAIL_BUDGET,
                )
                pending_user.append(
                    _tool_result_block(
                        main_id, f"{main_name}: task_success returned\n{rendered}"
                    )
                )
                pending_main = None

        elif isinstance(event, CancelledEvent):
            if pending_main is not None:
                main_id, main_name = pending_main
                msg = (
                    f"{main_name}: cancelled after "
                    f"{event.iterations_completed} iterations"
                )
                pending_user.append(_tool_result_block(main_id, msg))
                pending_main = None

        elif isinstance(event, FailEvent):
            if pending_main is not None:
                main_id, main_name = pending_main
                pending_user.append(
                    _tool_result_block(
                        main_id, f"{main_name}: task_fail: {event.message}"
                    )
                )
                pending_main = None

        elif isinstance(event, ClarifyEvent):
            if pending_main is not None:
                main_id, main_name = pending_main
                pending_user.append(
                    _tool_result_block(
                        main_id, f"{main_name}: task_clarify: {event.message}"
                    )
                )
                pending_main = None

        elif isinstance(event, FileEvent):
            pending_user.append({"type": "text", "text": _file_event_to_text(event)})

        elif isinstance(event, ChapterEvent):
            text, _ = render_chapter(event.name, event.message)
            pending_user.append({"type": "text", "text": text})

        elif isinstance(event, SystemNoteEvent):
            pending_user.append({"type": "text", "text": event.message})

    flush_user()
    return messages
