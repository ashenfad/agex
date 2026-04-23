"""
Common helpers, constants, and event factories for the task loop.

This module re-exports from focused sub-modules for backward compatibility.
New code should import directly from the sub-modules:
  - agex.agent.loop.state_helpers
  - agex.agent.loop.event_factories
  - agex.agent.loop.file_editing
"""

from __future__ import annotations

# --- Third-party / internal re-exports used by sync_loop, async_loop, mixin ---
from kvgit import Namespaced, Staged
from pydantic import ValidationError

from agex.agent.datatypes import (
    LLMFail,
    TaskCancelled,
    TaskClarify,
    TaskFail,
    TaskSuccess,
    TaskTimeout,
    _AgentExit,
)
from agex.agent.events import (
    ActionEvent,
    ClarifyEvent,
    ErrorEvent,
    FailEvent,
    OutputEvent,
    SuccessEvent,
    SystemNoteEvent,
    TaskStartEvent,
)
from agex.eval.error import EvalError
from agex.eval.objects import PrintAction
from agex.llm.core import (
    EmissionsBuilder,
    LLMResponse,
    ResponseBuilder,
    ResponseParseError,
    StreamToken,
)
from agex.state import (
    ConcurrencyError,
    MergeConflict,
    events,
    is_live_root,
    safe_commit,
)
from agex.state.live import Live
from agex.state.log import add_event_to_log, get_events_from_log

# --- Re-exports from sub-modules ---
from .event_factories import (
    TASK_CONTROL_GUIDANCE,
    create_action_event,
    create_clarify_event,
    create_error_output,
    create_fail_event,
    create_guidance_output,
    create_success_event,
    create_task_start_event,
    create_transient_event,
    create_unsaved_warning,
    execute_terminal,
    maybe_add_file_event,
    maybe_file_event,
    strip_namespace_prefix,
)
from .file_editing import (
    _adjust_replacement_indent,
    _build_trailing_ws_pattern,
    _find_indent_flexible_match,
    apply_file_edit,
    apply_file_write,
    apply_optimistic_file_actions,
)
from .state_helpers import (
    check_cancellation,
    check_for_task_call,
    initialize_exec_state,
    yield_new_events,
)

__all__ = [
    # Constants
    "TASK_CONTROL_GUIDANCE",
    # Event factories
    "create_task_start_event",
    "create_action_event",
    "create_success_event",
    "create_clarify_event",
    "create_fail_event",
    "create_error_output",
    "create_guidance_output",
    "create_transient_event",
    "create_unsaved_warning",
    # Terminal execution
    "execute_terminal",
    # State helpers
    "initialize_exec_state",
    "check_for_task_call",
    "strip_namespace_prefix",
    "yield_new_events",
    "maybe_file_event",
    "maybe_add_file_event",
    # File emission appliers
    "apply_file_write",
    "apply_file_edit",
    "apply_optimistic_file_actions",
    "safe_commit",
    "ResponseBuilder",
    "EmissionsBuilder",
    # Re-exports
    "ValidationError",
    "LLMFail",
    "TaskClarify",
    "TaskFail",
    "TaskSuccess",
    "TaskTimeout",
    "TaskCancelled",
    "_AgentExit",
    "ActionEvent",
    "ClarifyEvent",
    "ErrorEvent",
    "FailEvent",
    "OutputEvent",
    "SuccessEvent",
    "SystemNoteEvent",
    "TaskStartEvent",
    "EvalError",
    "PrintAction",
    "LLMResponse",
    "ResponseParseError",
    "StreamToken",
    "ConcurrencyError",
    "MergeConflict",
    "Live",
    "Namespaced",
    "Staged",
    "events",
    "is_live_root",
    "add_event_to_log",
    "get_events_from_log",
    "check_cancellation",
    # Private but used by tests
    "_build_trailing_ws_pattern",
    "_find_indent_flexible_match",
    "_adjust_replacement_indent",
]
