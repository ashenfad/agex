import os
import shutil
import sys
from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, TextIO, cast

from agex.llm.core import StreamToken

from ..eval.objects import PrintAction
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
_last_section_done: bool = True


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


def _summarize_output_parts(
    parts: list[Any], max_preview: int = 80, verbosity: str = "normal"
) -> str:
    if not parts:
        return "0 parts"
    # If just one part, try to give a succinct preview
    if len(parts) == 1:
        part = parts[0]
        type_name = type(part).__name__
        if isinstance(part, str):
            if verbosity == "verbose":
                content = _strip_newlines(part)
            else:
                content = _truncate(_strip_newlines(part), max_preview)
            return f"text {content}"
        if isinstance(part, PrintAction):
            # PrintAction is a tuple containing the printed arguments
            # Join them with spaces and show the content instead of just metadata
            content = " ".join(str(item) for item in part)
            stripped_content = content.strip()

            # In verbose mode, don't truncate print content
            if verbosity == "verbose":
                final_content = _strip_newlines(stripped_content)
            else:
                final_content = _truncate(
                    _strip_newlines(stripped_content), max_preview
                )

            # Check if this looks like an error message and format accordingly
            if any(
                keyword in stripped_content
                for keyword in ["Error:", "Exception:", "Traceback", "ERROR:"]
            ):
                return f"error: {final_content}"
            else:
                return f"print {final_content}"
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
        from agex.agent.emissions import (
            FileEditEmission,
            FileWriteEmission,
            PythonEmission,
            TerminalEmission,
            TextEmission,
            ThinkingEmission,
        )

        # Aggregate emission content for a one-liner-per-section view.
        thinking_pieces: list[str] = []
        text_pieces: list[str] = []
        file_summaries: list[str] = []
        terminal_pieces: list[str] = []
        code_pieces: list[str] = []
        has_python_emission = False
        for em in event.emissions:
            if isinstance(em, PythonEmission):
                has_python_emission = True
                if em.thinking:
                    thinking_pieces.append(em.thinking)
                if em.code:
                    code_pieces.append(em.code)
            elif isinstance(em, TerminalEmission):
                if em.thinking:
                    thinking_pieces.append(em.thinking)
                if em.commands:
                    terminal_pieces.append(em.commands)
            elif isinstance(em, ThinkingEmission):
                if em.text and not em.redacted:
                    thinking_pieces.append(em.text)
            elif isinstance(em, TextEmission):
                if em.text:
                    text_pieces.append(em.text)
            elif isinstance(em, FileWriteEmission):
                tag = "append" if em.mode == "append" else "write"
                file_summaries.append(f"`{em.path}` ({tag})")
            elif isinstance(em, FileEditEmission):
                scope = " match_all" if em.match_all else ""
                file_summaries.append(f"`{em.path}` (edit {em.operation}{scope})")

        # Only emit the Thinking line if there was actually thinking —
        # otherwise the label is noise.
        if thinking_pieces:
            thinking_raw = "\n".join(thinking_pieces)
            if verbosity == "verbose":
                thinking_text = _strip_newlines(thinking_raw)
            else:
                thinking_text = _truncate(
                    _strip_newlines(thinking_raw), 120 if verbosity != "brief" else 80
                )
            body_lines.append(_indent(detail_indent, f"Thinking: {thinking_text}"))

        if text_pieces:
            report_raw = "\n\n".join(text_pieces)
            if verbosity == "verbose":
                report_display = _strip_newlines(report_raw)
            else:
                report_display = _truncate(
                    _strip_newlines(report_raw),
                    120 if verbosity != "brief" else 80,
                )
            report_line = f"Report: {report_display}"
            if use_color:
                report_line = _colorize(use_color, _Colors.green, report_line)
            body_lines.append(_indent(detail_indent, report_line))

        if file_summaries:
            body_lines.append(
                _indent(detail_indent, f"Files: {', '.join(file_summaries)}")
            )

        if terminal_pieces:
            full_terminal = "\n".join(terminal_pieces)
            term_lines_total = full_terminal.count("\n") + 1
            if verbosity == "verbose":
                shown = full_terminal.splitlines()[:truncate_code_lines]
                body_lines.append(_indent(detail_indent, "Terminal:"))
                for line in shown:
                    dimmed = _colorize(use_color, _Colors.dim, line)
                    body_lines.append(_indent(detail_indent + 2, dimmed))
                if len(shown) < term_lines_total:
                    remaining = term_lines_total - len(shown)
                    more_word = "line" if remaining == 1 else "lines"
                    more_text = f"… (+{remaining} more {more_word})"
                    more_text = _colorize(use_color, _Colors.dim, more_text)
                    body_lines.append(_indent(detail_indent + 2, more_text))
            else:
                first_line = full_terminal.splitlines()[0] if full_terminal else ""
                extra = term_lines_total - 1
                suffix = f" (+{extra} more)" if extra > 0 else ""
                body_lines.append(
                    _indent(detail_indent, f"Terminal: {first_line}{suffix}")
                )

        # Code section only renders when the turn contained a
        # PythonEmission — suppressing a misleading "Code: 0 lines" on
        # terminal-only / file-only turns.
        if has_python_emission:
            full_code = "\n".join(code_pieces)
            code_lines_total = full_code.count("\n") + 1 if full_code else 0
            if verbosity == "verbose" and full_code:
                shown_lines = full_code.splitlines()[:truncate_code_lines]
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
                body_lines.append(
                    _indent(detail_indent, f"Code: {code_lines_total} lines")
                )

        # No visible body means the turn carried only redacted /
        # signature-only thinking parts (native-thinking providers
        # sometimes emit these — a signed reasoning step with nothing
        # user-visible).  Show a hint so the event isn't a silent
        # headline that looks like a rendering bug.
        if len(body_lines) == 0:
            has_redacted_thinking = any(
                isinstance(em, ThinkingEmission)
                and (em.redacted or (em.signature is not None and not em.text))
                for em in event.emissions
            )
            hint = (
                "(signed thinking only — no user-visible content)"
                if has_redacted_thinking
                else "(no emissions)"
            )
            dim_hint = _colorize(use_color, _Colors.dim, hint)
            body_lines.append(_indent(detail_indent, dim_hint))

    elif isinstance(event, OutputEvent):
        summary = _summarize_output_parts(event.parts, verbosity=verbosity)
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


def pprint_tokens(
    token: StreamToken,
    *,
    color: str = "auto",
    stream: TextIO | None = None,
) -> None:
    """
    Pretty-print a streamed token from on_token handler.

    Can be used directly as an on_token handler: task(on_token=pprint_tokens)

    Args:
        token: A TokenChunk from the LLM stream.
        color: "auto" | "always" | "never". Honors NO_COLOR env var.
        stream: Output stream, defaults to sys.stdout.
    """

    if stream is None:
        stream = sys.stdout
    output_stream: TextIO = cast(TextIO, stream)

    use_color = _should_color(color, output_stream)

    # Signature tokens are invisible metadata — one per action-tool
    # call, carrying only the provider signature bytes.  Drop them
    # entirely; their done=True flag must NOT trigger the generic
    # section-newline below, otherwise every action-tool call gains a
    # blank line of whitespace in the console.
    if token.type == "signature":
        return

    # Prebuilt emissions arrive as a single ``emission`` token.  For
    # file emissions the path + content already streamed via the
    # ``file_*`` tokens, so we only echo the mode/operation trailer
    # here (no duplication).  Native-provider thinking blocks (Gemini
    # 3 thought parts, Claude thinking content) render as a 💭 block
    # so the signed reasoning is visible.
    if token.type == "emission" and token.emission is not None:
        from agex.agent.emissions import (
            FileEditEmission,
            FileWriteEmission,
            ThinkingEmission,
        )

        em = token.emission
        if isinstance(em, FileWriteEmission):
            label = "append" if em.mode == "append" else "create"
            line = f"  → {label}\n"
            color_code = _Colors.magenta if use_color else ""
        elif isinstance(em, FileEditEmission):
            scope = " (match_all)" if em.match_all else ""
            line = f"  → {em.operation}{scope}\n"
            color_code = _Colors.magenta if use_color else ""
        elif isinstance(em, ThinkingEmission):
            if em.redacted:
                line = "💭 [redacted thinking]\n"
            elif em.text and em.text.strip():
                # Strip leading/trailing whitespace — native thought
                # parts often arrive with padding newlines that would
                # otherwise render as runs of blank lines between
                # sections.  Mirror the streaming ``thinking`` token's
                # colour so native thought parts don't visually split
                # from narration-via-schema thinking.
                line = f"💭 {em.text.strip()}\n"
            else:
                return  # Nothing worth printing (signature-only or whitespace).
            color_code = _Colors.bright_blue if use_color else ""
        else:
            line = f"emission: {em!r}\n"
            color_code = _Colors.magenta if use_color else ""
        if use_color and color_code:
            line = _colorize(use_color, color_code, line)
        output_stream.write(line)
        output_stream.flush()
        return

    if token.done:
        # Ensure sections end with a newline for readability (but only once)
        output_stream.write("\n")
        output_stream.flush()
        return

    # Print content with color
    content = token.content

    # Color based on token type and optionally prepend context
    prefix = ""
    if token.type == "title" and token.start:
        timestamp = (
            token.timestamp.astimezone().strftime("%H:%M:%S") if token.timestamp else ""
        )
        agent = token.agent_name or "agent"
        prefix = f"\n[{timestamp}] {agent}: " if timestamp else f"{agent}: "
        color_code = _Colors.cyan if use_color else ""
    elif token.type == "thinking":
        color_code = _Colors.bright_blue if use_color else ""
    elif token.type == "text":
        # User-facing prose — distinct label at section start so the
        # stream is visually separable from thinking and code.
        if token.start:
            prefix = "\n💬 "
        color_code = _Colors.green if use_color else ""
    elif token.type == "python":
        color_code = _Colors.yellow if use_color else ""
    elif token.type == "terminal":
        color_code = _Colors.green if use_color else ""
    elif token.type == "file_path":
        # One emoji label per file tool call.  Content streams inline
        # until done=True trips the generic newline branch above.
        if token.start:
            prefix = "\n📁 "
        color_code = _Colors.magenta if use_color else ""
    elif token.type == "file_search":
        if token.start:
            prefix = "🔍 "
        color_code = _Colors.yellow if use_color else ""
    elif token.type == "file_content":
        if token.start:
            prefix = "📝\n"
        color_code = _Colors.dim if use_color else ""
    else:
        color_code = _Colors.cyan if use_color else ""

    # Final content assembly
    if token.start and token.type in (
        "title",
        "text",
        "file_path",
        "file_search",
        "file_content",
    ):
        final_text = prefix + content
    else:
        final_text = content
    if use_color and color_code:
        final_text = _colorize(use_color, color_code, final_text)
    output_stream.write(final_text)
    output_stream.flush()
