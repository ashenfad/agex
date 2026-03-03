import html
from datetime import datetime, timezone
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, model_validator

from ..agent.datatypes import EditAction, FileAction
from ..eval.objects import ImageAction, PrintAction
from ..render.primitives import (
    HI_DETAIL_BUDGET,
    LOW_DETAIL_BUDGET,
    count_tokens,
    render_action_markdown,
    render_chapter,
    render_fail,
    render_output_parts_full,
    render_success,
    render_task_start,
)


def _suppress_stdout(func: Callable[[], Any]) -> Any:
    """
    Context manager helper to suppress stdout during function execution.

    Prevents Plotly figures and other objects with IPython display hooks from
    printing HTML to stdout in NiceGUI/IPython environments.
    """
    import sys
    from io import StringIO

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        return func()
    finally:
        sys.stdout = old_stdout


def _render_object_as_html(obj: Any) -> str:
    """
    Helper function to render an object as HTML with rich representation fallbacks.

    Tries in order:
    1. _repr_html_ method (pandas DataFrames, plotly Figures, etc.)
    2. _repr_mimebundle_ method (matplotlib figures, etc.)
    3. Escaped string representation as fallback

    Returns HTML string ready for inclusion in a larger HTML structure.
    """
    try:
        # Handle ImageAction objects explicitly (from view_image calls)
        if isinstance(obj, ImageAction):
            # Suppress stdout during _repr_html_ call to prevent Plotly figures
            # (and other objects with IPython display hooks) from printing HTML to stdout
            # in NiceGUI/IPython environments
            html_content = _suppress_stdout(lambda: obj._repr_html_())
            return f"<div style='margin: 5px 0;'>{html_content}</div>"
        # Handle PrintAction objects by joining their content
        elif isinstance(obj, PrintAction):
            # Join the PrintAction tuple content with spaces, like print() does
            content = " ".join(str(item) for item in obj)
            escaped_content = html.escape(content)

            # Check if this looks like an error and style accordingly
            if any(
                keyword in content
                for keyword in ["Error:", "Exception:", "Traceback", "ERROR:"]
            ):
                return f"<pre style='background: #fff2f0; padding: 8px; border-radius: 3px; margin: 0; color: #d73a49; font-family: monospace; border-left: 3px solid #d73a49;'>{escaped_content}</pre>"
            else:
                return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_content}</pre>"
        # Check if object has _repr_html_ method (pandas DataFrames, plotly Figures, etc.)
        elif hasattr(obj, "_repr_html_"):
            # For DataFrames, ensure we use pandas default display settings for HTML
            # (not the temporarily-modified settings used for LLM token optimization)
            from agex.render.primitives import is_dataframe

            if is_dataframe(obj):
                try:
                    import pandas as pd

                    # Save current settings
                    old_max_rows = pd.options.display.max_rows
                    old_min_rows = pd.options.display.min_rows

                    # Restore pandas defaults for HTML rendering
                    pd.options.display.max_rows = 60  # pandas default
                    pd.options.display.min_rows = 10  # pandas default

                    try:
                        html_content = _suppress_stdout(lambda: obj._repr_html_())
                        return f"<div style='margin: 5px 0;'>{html_content}</div>"
                    finally:
                        # Restore whatever settings were active
                        pd.options.display.max_rows = old_max_rows
                        pd.options.display.min_rows = old_min_rows
                except ImportError:
                    pass

            # Non-DataFrame or pandas not available: use regular rendering
            # Suppress stdout during _repr_html_ call to prevent Plotly figures
            # (and other objects with IPython display hooks) from printing HTML to stdout
            # in NiceGUI/IPython environments
            html_content = _suppress_stdout(lambda: obj._repr_html_())
            return f"<div style='margin: 5px 0;'>{html_content}</div>"
        # Check for _repr_mimebundle_ (matplotlib figures, etc.)
        elif hasattr(obj, "_repr_mimebundle_"):
            # Suppress stdout during _repr_mimebundle_ call to prevent Plotly figures
            # (and other objects with IPython display hooks) from printing HTML to stdout
            # in NiceGUI/IPython environments
            bundle = _suppress_stdout(
                lambda: obj._repr_mimebundle_(include=["text/html"])
            )
            if "text/html" in bundle:
                return f"<div style='margin: 5px 0;'>{bundle['text/html']}</div>"
            else:
                # Fall back to escaped string representation
                escaped_obj = html.escape(str(obj))
                return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_obj}</pre>"
        else:
            # Default to escaped string representation
            escaped_obj = html.escape(str(obj))
            return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_obj}</pre>"
    except Exception:
        # Fallback to string if anything goes wrong
        escaped_obj = html.escape(str(obj))
        return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_obj}</pre>"


def _event_html_container(
    emoji: str,
    event_type: str,
    full_namespace: str,
    timestamp: datetime,
    content: str,
    commit_hash: str | None = None,
) -> str:
    """
    Helper function to create consistent HTML containers for events.

    Args:
        emoji: The emoji icon for the event type
        event_type: The name of the event type (e.g., "ActionEvent")
        agent_name: The name of the agent
        timestamp: The datetime object to format
        content: The main content HTML
        commit_hash: Optional commit hash to display

    Returns:
        Complete HTML string for the event
    """
    # Format timestamp as succinct ISO format: 2025-07-23T20:53Z
    formatted_timestamp = (
        timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )

    # Build metadata line with timestamp and optional commit hash
    metadata_parts = [formatted_timestamp]
    if commit_hash:
        metadata_parts.append(
            f'<code style="background: #f1f3f4; padding: 1px 4px; border-radius: 2px; font-family: monospace; font-size: 11px;">{commit_hash[:8]}</code>'
        )

    metadata_line = " • ".join(metadata_parts)

    return f"""
    <div style="border: 1px solid #e1e4e8; border-radius: 8px; padding: 16px; margin: 8px 0; background: #f6f8fa;">
        <div style="font-weight: 600; color: #24292e; margin-bottom: 8px; font-size: 14px;">
            {emoji} {event_type} - {full_namespace}
        </div>
        <div style="font-size: 12px; color: #6a737d; margin-bottom: 12px;">
            {metadata_line}
        </div>
        {content}
    </div>
    """


def _event_section(title: str, content: str, color: str = "#6a737d") -> str:
    """
    Helper function to create consistent sections within events.

    Args:
        title: The section title
        content: The section content (HTML)
        color: The accent color for the section

    Returns:
        HTML string for the section
    """
    return f"""
    <div style="margin-bottom: 16px;">
        <div style="font-weight: 600; margin-bottom: 6px; color: {color}; font-size: 13px;">
            {title}
        </div>
        <div style="background: #fff; padding: 12px; border-radius: 6px; border-left: 3px solid {color}; color: #24292e; line-height: 1.4;">
            {content}
        </div>
    </div>
    """


def _code_section(title: str, code: str, color: str = "#28a745") -> str:
    """
    Helper function to create consistent code sections within events.

    Args:
        title: The section title
        code: The code content (will be syntax highlighted)
        color: The accent color for the section

    Returns:
        HTML string for the code section with syntax highlighting
    """
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer

        # Create a formatter with high-contrast styling (Xcode colors)
        formatter = HtmlFormatter(
            style="solarized-light", noclasses=True, nobackground=True, linenos=False
        )

        # Highlight the code
        highlighted_code = highlight(code, PythonLexer(), formatter)

        return f"""
        <div>
            <div style="font-weight: 600; margin-bottom: 6px; color: {color}; font-size: 13px;">
                {title}
            </div>
            <div style="background: #fff; padding: 16px; border-radius: 6px; border-left: 3px solid {color}; overflow-x: auto; font-family: 'SFMono-Regular', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace; font-size: 13px; line-height: 1.45; margin: 0;">{highlighted_code}</div>
        </div>
        """
    except ImportError:
        # Fallback to plain HTML escaping if pygments is not available
        escaped_code = html.escape(code)
        return f"""
        <div>
            <div style="font-weight: 600; margin-bottom: 6px; color: {color}; font-size: 13px;">
                {title}
            </div>
            <pre style="background: #fff; padding: 16px; border-radius: 6px; border-left: 3px solid {color}; overflow-x: auto; font-family: 'SFMono-Regular', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace; font-size: 13px; line-height: 1.45; margin: 0; color: #24292e;"><code>{escaped_code}</code></pre>
        </div>
        """


def _render_markdown_html(text: str) -> str:
    """
    Helper function to render markdown text to HTML with a fallback.

    Args:
        text: The markdown text to render

    Returns:
        HTML string
    """
    try:
        import markdown

        return markdown.markdown(
            text, extensions=["fenced_code", "nl2br", "sane_lists"]
        )
    except ImportError:
        # Fallback to escaped text wrapped in pre to preserve formatting
        return f"<pre style='white-space: pre-wrap; margin: 0; font-family: inherit;'>{html.escape(text)}</pre>"


class BaseEvent(BaseModel):
    """Base class for all agent events with common fields."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str
    full_namespace: str = ""  # Will be set by add_event_to_log
    commit_hash: str | None = None
    source: Literal["setup", "main"] = "main"  # Execution phase
    full_detail_tokens: int = 0  # Cached token estimate for full render
    low_detail_tokens: int = 0  # Cached token estimate for low-detail render

    def __repr_args__(self):
        """Override Pydantic's repr args to customize the display."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        class_name = self.__class__.__name__
        return [("event", f"{class_name}[{self.full_namespace}] @ {time_str}")]

    def __repr_str__(self, join_str: str) -> str:
        """Override Pydantic's repr string to use our custom format."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        class_name = self.__class__.__name__
        return f"{class_name}[{self.full_namespace}] @ {time_str}"

    def _repr_markdown_(self) -> str:
        """Rich markdown representation for notebook display."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        class_name = self.__class__.__name__

        # Map event types to emojis
        emoji_map = {
            "TaskStartEvent": "🚀",
            "ActionEvent": "🧠",
            "OutputEvent": "📤",
            "SuccessEvent": "✅",
            "FailEvent": "❌",
            "ClarifyEvent": "❓",
            "ChapterEvent": "📖",
            "ErrorEvent": "⚠️",
        }
        emoji = emoji_map.get(class_name, "📋")

        return f"## {emoji} {class_name} - {self.full_namespace}\n**Time:** {time_str}"

    def as_markdown(self) -> str:
        """Get the markdown representation for use outside notebooks."""
        return self._repr_markdown_()

    def __str__(self) -> str:
        """Detailed string representation for debugging."""
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        class_name = self.__class__.__name__
        return f"{class_name}[{self.full_namespace}] @ {time_str}"

    def __format__(self, format_spec: str) -> str:
        """Custom formatting support.

        Format specs:
        - 'markdown' or 'md': Return markdown representation
        - 'detailed' or 'd': Return detailed string with extra info
        - '' (empty): Return standard repr
        """
        if format_spec in ("markdown", "md"):
            return self._repr_markdown_()
        elif format_spec in ("detailed", "d"):
            return str(self)  # Use the detailed __str__ method
        else:
            return self.__repr_str__("")  # Use standard repr

    def _repr_html_(self) -> str:
        """Default to markdown but expect subclasses to override."""
        return self._repr_markdown_()

    def as_html(self) -> str:
        """Get the html representation for use outside notebooks."""
        return self._repr_html_()


class TaskStartEvent(BaseEvent):
    """Fired once at the beginning of a task."""

    task_name: str
    inputs: dict[str, Any]
    message: str  # The formatted task message for the LLM

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, hi_tokens = render_task_start(self.message, budget=HI_DETAIL_BUDGET)
        _, lo_tokens = render_task_start(self.message, budget=LOW_DETAIL_BUDGET)
        self.full_detail_tokens = hi_tokens
        self.low_detail_tokens = lo_tokens
        return self

    def __str__(self) -> str:
        """Detailed string with task information."""
        base = super().__str__()
        inputs_preview = str(self.inputs)
        if len(inputs_preview) > 60:
            inputs_preview = inputs_preview[:57] + "..."
        return f"{base}\n  Task: {self.task_name}\n  Inputs: {inputs_preview}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with task details."""
        base = super()._repr_markdown_()
        inputs_json = str(self.inputs).replace("'", '"')  # Quick JSON-ish format
        return f"""{base}  
**Task:** `{self.task_name}`  
**Inputs:**
```json
{inputs_json}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        # Create individual sections for each input, similar to OutputEvent parts
        content_sections = []

        for input_name, input_value in self.inputs.items():
            # Render each input using the same logic as OutputEvent parts
            input_html = _render_object_as_html(input_value)
            input_section = _event_section(f"📋 {input_name}:", input_html, "#6a737d")
            content_sections.append(input_section)

        # If no inputs, show a simple message
        if not content_sections:
            content_sections.append(
                _event_section("📋 Inputs:", "No inputs provided", "#6a737d")
            )

        content = "".join(content_sections)

        return _event_html_container(
            "🚀",
            "TaskStartEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class ActionEvent(BaseEvent):
    """Fired when the agent decides on its next thought and action (code or terminal)."""

    title: str = ""
    thinking: str
    code: str | None = None
    terminal: str | None = None
    file_actions: list[FileAction | EditAction] = Field(default_factory=list)
    input_tokens: int | None = None
    output_tokens: int | None = None

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, tokens = render_action_markdown(
            self.thinking, self.code, self.title, self.file_actions, self.terminal
        )
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens  # No separate low-detail rendering
        return self

    def __str__(self) -> str:
        """Detailed string with thinking and code/terminal preview."""
        base = super().__str__()
        title_text = f"\n  Title: {self.title}" if self.title else ""
        thinking_preview = (
            self.thinking[:80] + "..." if len(self.thinking) > 80 else self.thinking
        )

        files_info = []
        for action in self.file_actions:
            if isinstance(action, EditAction):
                match_all_str = ", match_all" if action.match_all else ""
                op_str = (
                    f", {action.operation}" if action.operation != "replace" else ""
                )
                files_info.append(f"{action.path} (edit{op_str}{match_all_str})")
            else:
                files_info.append(f"{action.path} ({action.mode})")

        files_text = f"\n  Files: {files_info}" if files_info else ""

        if self.terminal:
            terminal_lines = self.terminal.count("\n") + 1
            action_text = f"\n  Terminal: {terminal_lines} lines"
        else:
            code_lines = (self.code or "").count("\n") + 1
            action_text = f"\n  Code: {code_lines} lines"

        return f"{base}{title_text}\n  Thinking: {thinking_preview}{action_text}{files_text}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with code/terminal block."""
        base = super()._repr_markdown_()
        title_section = f"**Title:** {self.title}\n\n" if self.title else ""

        files_section = ""
        if self.file_actions:
            files_section = "**Files:**\n"
            for action in self.file_actions:
                if isinstance(action, EditAction):
                    match_all_str = ", match_all" if action.match_all else ""
                    op_str = (
                        f", {action.operation}" if action.operation != "replace" else ""
                    )
                    files_section += (
                        f" - `{action.path}` (edit{op_str}{match_all_str})\n"
                    )
                else:
                    mode_suffix = f" ({action.mode})" if action.mode != "write" else ""
                    files_section += f" - `{action.path}`{mode_suffix}\n"
            files_section += "\n"

        if self.terminal:
            action_section = f"""**Terminal:**
```bash
{self.terminal}
```"""
        else:
            action_section = f"""**Code:**
```python
{self.code or ""}
```"""

        return f"""{base}
{title_section}**Thinking:** {self.thinking}

{files_section}{action_section}"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        sections = []
        if self.title:
            sections.append(
                _event_section("📝 Title:", html.escape(self.title), "#6f42c1")
            )

        # Create the thinking section
        thinking_html = _render_markdown_html(self.thinking)
        thinking_section = _event_section("💭 Thinking:", thinking_html, "#0366d6")

        if self.file_actions:
            file_links_parts = []
            for action in self.file_actions:
                if isinstance(action, EditAction):
                    # Build combined label: (edit, insert-after, match_all)
                    label_parts = ["edit"]
                    if action.operation != "replace":
                        label_parts.append(action.operation)
                    if action.match_all:
                        label_parts.append("match_all")
                    label = ", ".join(label_parts)
                    file_links_parts.append(
                        f"<code>{action.path}</code> <small>({label})</small>"
                    )
                else:
                    mode_suffix = (
                        f" <small>({action.mode})</small>"
                        if action.mode != "write"
                        else ""
                    )
                    file_links_parts.append(f"<code>{action.path}</code>{mode_suffix}")

            file_links = ", ".join(file_links_parts)
            sections.append(_event_section("📁 Files:", file_links, "#28a745"))

        # Create the action section (terminal or code)
        if self.terminal:
            action_section = _code_section("💻 Terminal:", self.terminal, "#6f42c1")
        else:
            action_section = _code_section("🐍 Code:", self.code or "", "#28a745")

        sections.extend([thinking_section, action_section])

        content = "".join(sections)

        return _event_html_container(
            "🧠",
            "ActionEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class OutputEvent(BaseEvent):
    """A container for objects produced by the agent's code."""

    parts: list[Any]

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, hi_tokens = render_output_parts_full(self.parts, budget=HI_DETAIL_BUDGET)
        _, lo_tokens = render_output_parts_full(self.parts, budget=LOW_DETAIL_BUDGET)
        self.full_detail_tokens = hi_tokens
        self.low_detail_tokens = lo_tokens
        return self

    def __str__(self) -> str:
        """Detailed string with output summary."""
        base = super().__str__()
        parts_summary = f"{len(self.parts)} parts"
        if self.parts:
            first_part = str(self.parts[0])
            if len(first_part) > 40:
                first_part = first_part[:37] + "..."
            parts_summary += f" (first: {first_part})"
        return f"{base}\n  Output: {parts_summary}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with output display."""
        base = super()._repr_markdown_()
        output_md = "\n**Output:**\n"
        for part in self.parts:
            output_md += f"```\n{part}\n```\n"
        return base + output_md

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        # Add each output part using the helper function
        parts_html = ""
        for part in self.parts:
            parts_html += _render_object_as_html(part)

        content = _event_section("📤 Output:", parts_html, "#6f42c1")

        return _event_html_container(
            "🤖",
            "OutputEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class ErrorEvent(BaseEvent):
    """Fired for framework-level errors that agents shouldn't need to handle."""

    error: Any  # The actual exception object
    recoverable: bool = True  # Whether the task can continue after this error

    @model_validator(mode="after")
    def _compute_tokens(self):
        # ErrorEvents are not shown to agents, but compute tokens for consistency
        tokens = count_tokens(str(self.error))
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens  # No separate low-detail rendering
        return self

    def __str__(self) -> str:
        """Detailed string with error information."""
        base = super().__str__()
        error_name = (
            type(self.error).__name__
            if hasattr(self.error, "__class__")
            else str(type(self.error))
        )
        error_msg = (
            str(self.error)[:60] + "..."
            if len(str(self.error)) > 60
            else str(self.error)
        )
        status = "recoverable" if self.recoverable else "fatal"
        return f"{base}\n  Error: {error_name}: {error_msg} ({status})"

    def _repr_markdown_(self) -> str:
        """Rich markdown with error details."""
        base = super()._repr_markdown_()
        error_name = (
            type(self.error).__name__
            if hasattr(self.error, "__class__")
            else str(type(self.error))
        )
        status = "🔄 Recoverable" if self.recoverable else "💀 Fatal"
        return f"""{base}  
**Error:** `{error_name}`  
**Status:** {status}  
**Message:**
```
{self.error}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        error_name = (
            type(self.error).__name__
            if hasattr(self.error, "__class__")
            else str(type(self.error))
        )
        status = "🔄 Recoverable" if self.recoverable else "💀 Fatal"
        status_color = "#fb8500" if self.recoverable else "#d73a49"

        error_section = _event_section(
            f"⚠️ Error: `{error_name}`", html.escape(str(self.error)), "#d73a49"
        )
        status_section = _event_section("📊 Status:", status, status_color)

        content = error_section + status_section

        return _event_html_container(
            "🚨",
            "ErrorEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class SuccessEvent(BaseEvent):
    """Fired when the task completes successfully."""

    result: Any

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, hi_tokens = render_success(self.result, budget=HI_DETAIL_BUDGET)
        _, lo_tokens = render_success(self.result, budget=LOW_DETAIL_BUDGET)
        self.full_detail_tokens = hi_tokens
        self.low_detail_tokens = lo_tokens
        return self

    def __str__(self) -> str:
        """Detailed string with result preview."""
        base = super().__str__()
        result_preview = (
            str(self.result)[:60] + "..."
            if len(str(self.result)) > 60
            else str(self.result)
        )
        return f"{base}\n  Result: {result_preview}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with result display."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Result:**
```
{self.result}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        # Render the result using the helper function
        result_html = _render_object_as_html(self.result)
        content = _event_section("✨ Result:", result_html, "#28a745")

        return _event_html_container(
            "✅",
            "SuccessEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class FailEvent(BaseEvent):
    """Fired when the task is explicitly failed."""

    message: str

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, tokens = render_fail(self.message)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens  # No separate low-detail rendering
        return self

    def __str__(self) -> str:
        """Detailed string with failure message."""
        base = super().__str__()
        return f"{base}\n  Message: {self.message}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with failure details."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Failure Message:**
```
{self.message}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        content = _event_section(
            "💥 Failure Message:", html.escape(self.message), "#d73a49"
        )

        return _event_html_container(
            "❌",
            "FailEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class CancelledEvent(BaseEvent):
    """Fired when the task is cancelled via external request."""

    task_name: str
    iterations_completed: int

    @model_validator(mode="after")
    def _compute_tokens(self):
        text = f"🛑 Task '{self.task_name}' cancelled after {self.iterations_completed} iterations"
        tokens = count_tokens(text)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens  # No separate low-detail rendering
        return self

    def __str__(self) -> str:
        """Detailed string with cancellation info."""
        base = super().__str__()
        return f"{base}\n  Task: {self.task_name}\n  Iterations: {self.iterations_completed}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with cancellation details."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Task:** `{self.task_name}`  
**Iterations completed:** {self.iterations_completed}"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        content = _event_section(
            f"🛑 Task `{self.task_name}` cancelled",
            f"Completed {self.iterations_completed} iteration(s) before cancellation",
            "#fb8500",
        )

        return _event_html_container(
            "🛑",
            "CancelledEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class ClarifyEvent(BaseEvent):
    """Fired when the task is paused for clarification."""

    message: str

    @model_validator(mode="after")
    def _compute_tokens(self):
        # Clarify is similar to fail in rendering
        text = f"❓ Clarification needed: {self.message}"
        tokens = count_tokens(text)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens  # No separate low-detail rendering
        return self

    def __str__(self) -> str:
        """Detailed string with clarification message."""
        base = super().__str__()
        return f"{base}\n  Message: {self.message}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with clarification details."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Clarification Request:**
```
{self.message}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        content = _event_section(
            "❓ Clarification Request:", html.escape(self.message), "#fb8500"
        )

        return _event_html_container(
            "🤔",
            "ClarifyEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class SystemNoteEvent(BaseEvent):
    """Fired when the system injects a transient note (e.g. iteration warning)."""

    message: str

    @model_validator(mode="after")
    def _compute_tokens(self):
        tokens = count_tokens(self.message)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens
        return self

    def __str__(self) -> str:
        """Detailed string."""
        base = super().__str__()
        return f"{base}\n  Message: {self.message}"

    def _repr_markdown_(self) -> str:
        """Rich markdown."""
        base = super()._repr_markdown_()
        return f"""{base}  
**System Note:**
```
{self.message}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML."""
        content = _event_section(
            "🔔 System Note:", html.escape(self.message), "#0366d6"
        )
        return _event_html_container(
            "🔔",
            "SystemNoteEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class ChapterEvent(BaseEvent):
    """A named chapter that replaces a contiguous range of events.

    The original events are preserved inside the chapter (lossless).
    Only the chapter's name and message are rendered in context;
    the embedded events can be browsed via the /chapters VFS overlay.
    """

    name: str  # Short descriptive name
    message: str  # Agent's summary/distillation
    events: list[BaseEvent] = Field(default_factory=list)  # Original events (nested)

    @model_validator(mode="after")
    def _compute_tokens(self):
        _, tokens = render_chapter(self.name, self.message)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens
        return self

    def __str__(self) -> str:
        base = super().__str__()
        return f'{base}\n  Chapter: "{self.name}" ({len(self.events)} events)'

    def _repr_markdown_(self) -> str:
        base = super()._repr_markdown_()
        return f'{base}\n**Chapter:** "{self.name}" ({len(self.events)} events)\n\n{self.message}'

    def _repr_html_(self) -> str:
        header = f'📖 Chapter: "{self.name}" ({len(self.events)} events)'
        summary_html = _render_markdown_html(self.message)
        content = _event_section(header, summary_html, "#6a737d")
        return _event_html_container(
            "📖",
            "ChapterEvent",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


class FileEvent(BaseEvent):
    """Fired when files are added, modified, or removed.

    Emitted after external API calls (source="user") or at end of agent
    turns (source="agent") when files have changed.
    """

    file_source: Literal[
        "user", "agent"
    ]  # Renamed to avoid conflict with BaseEvent.source
    added: list[str] = []
    modified: list[str] = []
    removed: list[str] = []

    @model_validator(mode="after")
    def _compute_tokens(self):
        # Build summary text for token counting
        parts = []
        if self.added:
            parts.append(f"Added: {', '.join(self.added)}")
        if self.modified:
            parts.append(f"Modified: {', '.join(self.modified)}")
        if self.removed:
            parts.append(f"Removed: {', '.join(self.removed)}")
        text = f"[File changes by {self.file_source}] " + "; ".join(parts)
        tokens = count_tokens(text)
        self.full_detail_tokens = tokens
        self.low_detail_tokens = tokens
        return self

    def __str__(self) -> str:
        """Detailed string with file change summary."""
        base = super().__str__()
        counts = []
        if self.added:
            counts.append(f"{len(self.added)} added")
        if self.modified:
            counts.append(f"{len(self.modified)} modified")
        if self.removed:
            counts.append(f"{len(self.removed)} removed")
        summary = ", ".join(counts) if counts else "no changes"
        return f"{base}\n  Source: {self.file_source}\n  Files: {summary}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with file lists."""
        base = super()._repr_markdown_()
        sections = []
        if self.added:
            sections.append(f"**Added:** {', '.join(self.added)}")
        if self.modified:
            sections.append(f"**Modified:** {', '.join(self.modified)}")
        if self.removed:
            sections.append(f"**Removed:** {', '.join(self.removed)}")
        content = "\n".join(sections) if sections else "*No file changes*"
        return f"{base}\n**Source:** {self.file_source}\n\n{content}"

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        sections = []
        if self.added:
            files_list = ", ".join(f"`{f}`" for f in self.added)
            sections.append(_event_section("📁 Added:", files_list, "#28a745"))
        if self.modified:
            files_list = ", ".join(f"`{f}`" for f in self.modified)
            sections.append(_event_section("✏️ Modified:", files_list, "#0366d6"))
        if self.removed:
            files_list = ", ".join(f"`{f}`" for f in self.removed)
            sections.append(_event_section("🗑️ Removed:", files_list, "#d73a49"))

        if not sections:
            sections.append(_event_section("📂 Files:", "No changes", "#6a737d"))

        content = "".join(sections)

        return _event_html_container(
            "📁",
            f"FileEvent ({self.file_source})",
            self.full_namespace,
            self.timestamp,
            content,
            self.commit_hash,
        )


Event = (
    TaskStartEvent
    | ActionEvent
    | OutputEvent
    | ErrorEvent
    | SuccessEvent
    | FailEvent
    | CancelledEvent
    | ClarifyEvent
    | ChapterEvent
    | FileEvent
)


def _register_ipython_formatters():
    """
    Conditionally register rich IPython formatters if IPython is available.
    This enhances the display of OutputEvent in Jupyter notebooks without
    requiring IPython as a dependency.
    """
    try:
        from IPython.core.getipython import get_ipython

        # Only register if we're actually in an IPython environment
        ip = get_ipython()
        if ip is not None:
            # Register the HTML formatter for OutputEvent
            # This will use our _repr_html_ method automatically
            html_formatter = ip.display_formatter.formatters["text/html"]  # type: ignore[attr-defined]

            # Custom formatter function that uses our _repr_html_ method
            def as_html(obj):
                return obj._repr_html_()

            # Register the formatter
            html_formatter.for_type(OutputEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(SuccessEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(ActionEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(TaskStartEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(ErrorEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(FailEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(ClarifyEvent, as_html)  # type: ignore[attr-defined]
            html_formatter.for_type(ChapterEvent, as_html)  # type: ignore[attr-defined]

    except ImportError:
        # IPython not available - that's fine, we'll use the default _repr_markdown_
        pass
    except Exception:
        # Any other error in registration - fail silently and use defaults
        pass


# Register formatters when module is imported
_register_ipython_formatters()
