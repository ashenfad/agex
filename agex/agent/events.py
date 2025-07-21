from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class BaseEvent(BaseModel):
    """Base class for all agent events with common fields."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    agent_name: str

    def __repr_args__(self):
        """Override Pydantic's repr args to customize the display."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        class_name = self.__class__.__name__
        return [("event", f"{class_name}[{self.agent_name}] @ {time_str}")]

    def __repr_str__(self, join_str: str) -> str:
        """Override Pydantic's repr string to use our custom format."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        class_name = self.__class__.__name__
        return f"{class_name}[{self.agent_name}] @ {time_str}"

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
            "ErrorEvent": "⚠️",
        }
        emoji = emoji_map.get(class_name, "📋")

        return f"## {emoji} {class_name} - {self.agent_name}\n**Time:** {time_str}"

    def as_markdown(self) -> str:
        """Get the markdown representation for use outside notebooks."""
        return self._repr_markdown_()

    def __str__(self) -> str:
        """Detailed string representation for debugging."""
        time_str = self.timestamp.strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        class_name = self.__class__.__name__
        return f"{class_name}[{self.agent_name}] @ {time_str}"

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
        for i, part in enumerate(self.parts):
            output_md += f"```\n{part}\n```\n"
            if i >= 2:  # Limit display to first 3 parts
                output_md += f"... and {len(self.parts) - 3} more parts\n"
                break
        return base + output_md


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


class FailEvent(BaseEvent):
    """Fired when the task is explicitly failed."""

    message: str

    def __str__(self) -> str:
        """Detailed string with failure message."""
        base = super().__str__()
        message_preview = (
            self.message[:80] + "..." if len(self.message) > 80 else self.message
        )
        return f"{base}\n  Message: {message_preview}"

    def _repr_markdown_(self) -> str:
        """Rich markdown with failure details."""
        base = super()._repr_markdown_()
        return f"""{base}  
**Failure Message:**
```
{self.message}
```"""


Event = (
    TaskStartEvent | ActionEvent | OutputEvent | ErrorEvent | SuccessEvent | FailEvent
)
