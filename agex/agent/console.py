import os
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, TextIO, cast

from .events import (
    ActionEvent,
    BaseEvent,
    ClarifyEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    TaskStartEvent,
)

_last_timestamp: datetime | None = None


def _is_tty(stream: TextIO) -> bool:
    try:
        return stream.isatty()  # type: ignore[attr-defined]
    except Exception:
        return False


def _should_color(color: str, stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if color == "always":
        return True
    if color == "never":
        return False
    # auto
    return _is_tty(stream)


def _term_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 100


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 1:
        return text[:max_chars]
    return text[: max_chars - 1] + "…"


def _strip_newlines(text: str) -> str:
    """Replaces newline characters with spaces."""
    return text.replace("\n", " ")


def _short_hash(commit_hash: str | None) -> str:
    if not commit_hash:
        return ""
    return commit_hash[:8]


def _namespace_depth(full_namespace: str) -> int:
    if not full_namespace:
        return 0
    return full_namespace.count("/")


def _indent(prefix_spaces: int, text: str) -> str:
    if not text:
        return ""
    indent_str = " " * prefix_spaces
    return "\n".join(indent_str + line if line else line for line in text.splitlines())


def _format_timestamp(ts: datetime) -> str:
    # Show local time HH:MM:SS
    try:
        return ts.astimezone().strftime("%H:%M:%S")
    except Exception:
        return ts.strftime("%H:%M:%S")


def _format_delta(prev: datetime | None, current: datetime) -> str:
    if prev is None:
        return "Δ 0.0s"
    try:
        seconds = (current - prev).total_seconds()
        return f"Δ {seconds:.1f}s"
    except Exception:
        return "Δ ?"


class _Colors:
    reset = "\x1b[0m"
    bold = "\x1b[1m"
    dim = "\x1b[2m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    bright_blue = "\x1b[94m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"


def _colorize(use_color: bool, color_code: str, text: str) -> str:
    if not use_color:
        return text
    return f"{color_code}{text}{_Colors.reset}"


def _summarize_value(value: Any, max_chars: int = 60) -> str:
    try:
        # numpy/pandas shapes
        shape = getattr(value, "shape", None)
        if shape is not None:
            type_name = type(value).__name__
            return f"{type_name} shape={shape}"

        # len-able containers
        try:
            length = len(value)  # type: ignore[arg-type]
            type_name = type(value).__name__
            if isinstance(value, (str, bytes)):
                return _truncate(repr(value), max_chars)
            return f"{type_name} len={length}"
        except Exception:
            pass

        # fallback repr
        return _truncate(repr(value), max_chars)
    except Exception:
        return "<unrepr-able>"


def _summarize_inputs(
    inputs: dict[str, Any], max_items: int = 6, value_chars: int = 30
) -> str:
    if not inputs:
        return "(no inputs)"
    items: list[str] = []
    for index, (key, val) in enumerate(inputs.items()):
        if index >= max_items:
            items.append("…")
            break
        items.append(f"{key}={_truncate(_summarize_value(val), value_chars)}")
    return ", ".join(items)


def _summarize_output_parts(parts: list[Any], max_preview: int = 80) -> str:
    if not parts:
        return "0 parts"
    # If just one part, try to give a succinct preview
    if len(parts) == 1:
        part = parts[0]
        type_name = type(part).__name__
        if isinstance(part, str):
            return f"text {_truncate(_strip_newlines(part), max_preview)}"
        shape = getattr(part, "shape", None)
        if shape is not None:
            return f"{type_name} shape={shape}"
        try:
            length = len(part)  # type: ignore[arg-type]
            return f"{type_name} len={length}"
        except Exception:
            return type_name
    # Otherwise, count by type
    counts: dict[str, int] = {}
    for p in parts:
        counts[type(p).__name__] = counts.get(type(p).__name__, 0) + 1
    return ", ".join(f"{t} x{n}" for t, n in counts.items())


def _format_event_lines(
    event: BaseEvent,
    *,
    verbosity: Literal["brief", "normal", "verbose"],
    use_color: bool,
    indent_by_namespace: bool,
    truncate_code_lines: int,
    term_width: int,
) -> list[str]:
    # Header line parts
    emoji_map = {
        TaskStartEvent: "🚀",
        ActionEvent: "🧠",
        OutputEvent: "📤",
        SuccessEvent: "✅",
        FailEvent: "❌",
        ClarifyEvent: "❓",
        ErrorEvent: "⚠️",
    }
    color_map = {
        TaskStartEvent: _Colors.cyan,
        ActionEvent: _Colors.yellow,
        OutputEvent: _Colors.bright_blue,
        SuccessEvent: _Colors.green,
        FailEvent: _Colors.red,
        ClarifyEvent: _Colors.magenta,
        ErrorEvent: _Colors.red,
    }

    emoji = emoji_map.get(type(event), "📋")
    color_code = color_map.get(type(event), _Colors.cyan)

    timestamp_text = _format_timestamp(event.timestamp)
    id_text = f"{type(event).__name__} [{event.full_namespace}]"
    if event.commit_hash:
        id_text += f" @{_short_hash(event.commit_hash)}"

    header_depth = _namespace_depth(event.full_namespace) if indent_by_namespace else 0
    padding = header_depth * 3
    header_prefix = ("-" * padding + " ") if header_depth > 0 else ""

    header = f"{header_prefix}{timestamp_text}  {emoji} {id_text}"
    header = _colorize(use_color, color_code, header)

    # Body lines depend on event type
    body_lines: list[str] = []
    detail_indent = padding + 3

    if isinstance(event, TaskStartEvent):
        body_lines.append(_indent(detail_indent, f"Task: {event.task_name}"))
        preview = _summarize_inputs(event.inputs)
        body_lines.append(_indent(detail_indent, f"Inputs: {preview}"))

    elif isinstance(event, ActionEvent):
        thinking_preview = _truncate(
            _strip_newlines(event.thinking), 120 if verbosity != "brief" else 80
        )
        body_lines.append(_indent(detail_indent, f"Thinking: {thinking_preview}"))
        code_lines_total = event.code.count("\n") + 1 if event.code else 0
        if verbosity == "verbose" and event.code:
            shown_lines = event.code.splitlines()[:truncate_code_lines]
            for line in shown_lines:
                dimmed = _colorize(use_color, _Colors.dim, line)
                body_lines.append(_indent(detail_indent, dimmed))
            if len(shown_lines) < code_lines_total:
                remaining = max(code_lines_total - len(shown_lines), 0)
                more_word = "line" if remaining == 1 else "lines"
                more_text = f"… (+{remaining} more {more_word})"
                more_text = _colorize(use_color, _Colors.dim, more_text)
                body_lines.append(_indent(detail_indent, more_text))
        else:
            body_lines.append(_indent(detail_indent, f"Code: {code_lines_total} lines"))

    elif isinstance(event, OutputEvent):
        summary = _summarize_output_parts(event.parts)
        body_lines.append(_indent(detail_indent, f"Output: {summary}"))

    elif isinstance(event, SuccessEvent):
        result_summary = _summarize_value(event.result)
        body_lines.append(_indent(detail_indent, f"Result: {result_summary}"))

    elif isinstance(event, ErrorEvent):
        name = (
            type(event.error).__name__
            if hasattr(event.error, "__class__")
            else str(type(event.error))
        )
        msg = _truncate(
            _strip_newlines(str(event.error)), 160 if verbosity == "verbose" else 100
        )
        status = "recoverable" if event.recoverable else "fatal"
        body_lines.append(_indent(detail_indent, f"Error: {name}: {msg} ({status})"))

    elif isinstance(event, FailEvent):
        body_lines.append(
            _indent(
                detail_indent,
                f"Message: {_truncate(_strip_newlines(event.message), 160)}",
            )
        )

    elif isinstance(event, ClarifyEvent):
        body_lines.append(
            _indent(
                detail_indent,
                f"Message: {_truncate(_strip_newlines(event.message), 160)}",
            )
        )

    # Truncate hard if width is small
    if term_width and term_width > 20:
        max_len = term_width
        header = _truncate(header, max_len)
        body_lines = [_truncate(line, max_len) for line in body_lines]

    return [header] + body_lines


def pprint_events(
    events: BaseEvent | Iterable[BaseEvent],
    *,
    verbosity: Literal["brief", "normal", "verbose"] = "verbose",
    color: str = "auto",
    show_delta: bool = True,
    width: int | None = None,
    stream: TextIO | None = None,
    indent_by_namespace: bool = True,
    truncate_code_lines: int = 8,
) -> None:
    """
    Pretty-print one or more agent events to a console-like stream.

    Can be used directly as an on_event handler: agent.run_task(..., on_event=pprint_events)

    Args:
        events: A single event or an iterable of events.
        verbosity: "brief" | "normal" | "verbose".
        color: "auto" | "always" | "never". Honors NO_COLOR env var.
        show_delta: If True, appends time delta since the last printed event.
        width: Hard limit on line width. If None, uses terminal width.
        stream: Output stream, defaults to sys.stdout.
        indent_by_namespace: Indent details by hierarchical namespace depth.
        truncate_code_lines: In verbose mode, how many code lines to show.
    """

    global _last_timestamp

    if stream is None:
        stream = sys.stdout
    output_stream: TextIO = cast(TextIO, stream)

    use_color = _should_color(color, output_stream)
    term_width = width if width is not None else _term_width()

    # Normalize to iterable of events
    if isinstance(events, BaseEvent):
        iterator: Iterable[BaseEvent] = (events,)
    else:
        iterator = events

    for ev in iterator:
        lines = _format_event_lines(
            ev,
            verbosity=verbosity,
            use_color=use_color,
            indent_by_namespace=indent_by_namespace,
            truncate_code_lines=truncate_code_lines,
            term_width=term_width,
        )

        # Append delta to header if requested
        if show_delta and lines:
            delta_text = _format_delta(_last_timestamp, ev.timestamp)
            # Append as dim text at the end of the header
            if use_color:
                lines[0] = f"{lines[0]}  {_colorize(True, _Colors.dim, delta_text)}"
            else:
                lines[0] = f"{lines[0]}  {delta_text}"

        # Write
        for line in lines:
            try:
                output_stream.write(line + "\n")
            except Exception:
                # Best-effort; avoid crashing user code due to stream errors
                pass
        try:
            output_stream.flush()
        except Exception:
            pass

        _last_timestamp = ev.timestamp
