from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


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
        # Check if object has _repr_html_ method (pandas DataFrames, plotly Figures, etc.)
        if hasattr(obj, "_repr_html_"):
            return f"<div style='margin: 5px 0;'>{obj._repr_html_()}</div>"
        # Check for _repr_mimebundle_ (matplotlib figures, etc.)
        elif hasattr(obj, "_repr_mimebundle_"):
            bundle = obj._repr_mimebundle_(include=["text/html"])
            if "text/html" in bundle:
                return f"<div style='margin: 5px 0;'>{bundle['text/html']}</div>"
            else:
                # Fall back to escaped string representation
                import html

                escaped_obj = html.escape(str(obj))
                return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_obj}</pre>"
        else:
            # Default to escaped string representation
            import html

            escaped_obj = html.escape(str(obj))
            return f"<pre style='background: #fff; padding: 8px; border-radius: 3px; margin: 0; color: #24292e; font-family: monospace;'>{escaped_obj}</pre>"
    except Exception:
        # Fallback to string if anything goes wrong
        import html

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
        import html

        escaped_code = html.escape(code)

        return f"""
        <div>
            <div style="font-weight: 600; margin-bottom: 6px; color: {color}; font-size: 13px;">
                {title}
            </div>
            <pre style="background: #fff; padding: 16px; border-radius: 6px; border-left: 3px solid {color}; overflow-x: auto; font-family: 'SFMono-Regular', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace; font-size: 13px; line-height: 1.45; margin: 0; color: #24292e;"><code>{escaped_code}</code></pre>
        </div>
        """


class BaseEvent(BaseModel):
    """Base class for all agent events with common fields."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: str
    full_namespace: str = ""  # Will be set by add_event_to_log
    commit_hash: str | None = None

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


class TaskStartEvent(BaseEvent):
    """Fired once at the beginning of a task."""

    task_name: str
    inputs: dict[str, Any]
    message: str  # The formatted task message for the LLM

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
    """Fired when the agent decides on its next thought and code."""

    thinking: str
    code: str

    def __str__(self) -> str:
        """Detailed string with thinking and code preview."""
        base = super().__str__()
        thinking_preview = (
            self.thinking[:80] + "..." if len(self.thinking) > 80 else self.thinking
        )
        code_lines = self.code.count("\n") + 1
        return f"{base}\n  Thinking: {thinking_preview}\n  Code: {code_lines} lines"

    def _repr_markdown_(self) -> str:
        """Rich markdown with code block."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Thinking:** {self.thinking}

**Code:**
```python
{self.code}
```"""

    def _repr_html_(self) -> str:
        """Rich HTML representation for IPython/Jupyter environments."""
        import html

        # Create the thinking and code sections
        thinking_section = _event_section(
            "💭 Thinking:", html.escape(self.thinking), "#0366d6"
        )
        code_section = _code_section("🐍 Code:", self.code, "#28a745")

        content = thinking_section + code_section

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
        import html

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
        import html

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


class ClarifyEvent(BaseEvent):
    """Fired when the task is paused for clarification."""

    message: str

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
        import html

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


Event = (
    TaskStartEvent
    | ActionEvent
    | OutputEvent
    | ErrorEvent
    | SuccessEvent
    | FailEvent
    | ClarifyEvent
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

    except ImportError:
        # IPython not available - that's fine, we'll use the default _repr_markdown_
        pass
    except Exception:
        # Any other error in registration - fail silently and use defaults
        pass


# Register formatters when module is imported
_register_ipython_formatters()
