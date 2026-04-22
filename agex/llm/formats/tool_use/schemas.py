"""Tool schemas for the agex tool-use wire format.

Four tools cover what the XML format expressed as tags:

- ``python_action`` — run Python code (task-progress via ``task_success`` /
  ``task_fail`` / ``task_clarify`` / ``task_continue`` calls in the code).
- ``terminal_action`` — run shell commands (implicit ``task_continue``).
- ``write_file`` — write or append a file.
- ``edit_file`` — surgical edit with search + replace / insert-after /
  insert-before.

Each schema is returned as a provider-agnostic dict. Clients translate to
their provider's exact shape (OpenAI ``function.parameters``, Anthropic
``input_schema``, Gemini ``parameters``).
"""

from typing import Any

# Tool names — constants so the parser and renderer agree.
TOOL_PYTHON = "python_action"
TOOL_TERMINAL = "terminal_action"
TOOL_WRITE_FILE = "write_file"
TOOL_EDIT_FILE = "edit_file"

ACTION_TOOLS = frozenset({TOOL_PYTHON, TOOL_TERMINAL})
FILE_TOOLS = frozenset({TOOL_WRITE_FILE, TOOL_EDIT_FILE})
ALL_TOOLS = ACTION_TOOLS | FILE_TOOLS


_PYTHON_SCHEMA: dict[str, Any] = {
    "name": TOOL_PYTHON,
    "description": (
        "Run Python code. The task is driven by special calls inside the "
        "code: task_success(result) finishes successfully, task_fail(msg) "
        "finishes with an error, task_clarify(prompt) asks the caller a "
        "question, and task_continue() advances to the next turn so you can "
        "inspect output."
    ),
    "parameters": {
        "type": "object",
        "required": ["title", "thinking", "code"],
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for this turn (one line).",
            },
            "thinking": {
                "type": "string",
                "description": "Step-by-step reasoning for this turn.",
            },
            "report": {
                "type": "string",
                "description": (
                    "Optional short message for the user. Use on multi-turn "
                    "tasks to narrate slow steps; omit on trivial turns."
                ),
            },
            "code": {
                "type": "string",
                "description": "Python source to execute.",
            },
        },
    },
}


_TERMINAL_SCHEMA: dict[str, Any] = {
    "name": TOOL_TERMINAL,
    "description": (
        "Run shell commands. Implicitly continues the task — use "
        "python_action with task_success() / task_fail() to finish."
    ),
    "parameters": {
        "type": "object",
        "required": ["title", "thinking", "commands"],
        "properties": {
            "title": {
                "type": "string",
                "description": "Short title for this turn (one line).",
            },
            "thinking": {
                "type": "string",
                "description": "Step-by-step reasoning for this turn.",
            },
            "report": {
                "type": "string",
                "description": "Optional short message for the user.",
            },
            "commands": {
                "type": "string",
                "description": (
                    "Shell commands to run. Supported: ls, cat (with "
                    "-A/-n), head, tail, grep, find, wc, sort, uniq, cut, "
                    "diff, jq, cp, mv, rm, mkdir, touch, pwd, cd, echo, "
                    "tee, tar, gzip, gunzip, zip, unzip."
                ),
            },
        },
    },
}


_WRITE_FILE_SCHEMA: dict[str, Any] = {
    "name": TOOL_WRITE_FILE,
    "description": ("Write or append a file. Place Python modules under /helpers."),
    "parameters": {
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path within the agent's VFS.",
            },
            "content": {
                "type": "string",
                "description": "File contents to write.",
            },
            "mode": {
                "type": "string",
                "enum": ["write", "append"],
                "description": "Defaults to 'write'.",
            },
        },
    },
}


_EDIT_FILE_SCHEMA: dict[str, Any] = {
    "name": TOOL_EDIT_FILE,
    "description": (
        "Surgical edit. 'search' must match the file exactly (including "
        "whitespace) and occur once unless match_all=true. Provide EXACTLY "
        "one of 'replace', 'insert_after', or 'insert_before'. Prefer "
        "insert_after/insert_before over a replace that repeats the search "
        "text — the latter makes duplicates more likely if re-run."
    ),
    "parameters": {
        "type": "object",
        "required": ["path", "search"],
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute path within the agent's VFS.",
            },
            "search": {
                "type": "string",
                "description": ("Exact text to locate. Whitespace is significant."),
            },
            "replace": {
                "type": "string",
                "description": "Replacement text. Replaces 'search' entirely.",
            },
            "insert_after": {
                "type": "string",
                "description": "Text inserted after 'search' (kept verbatim).",
            },
            "insert_before": {
                "type": "string",
                "description": "Text inserted before 'search' (kept verbatim).",
            },
            "match_all": {
                "type": "boolean",
                "description": (
                    "If true, apply to every occurrence. Defaults to false."
                ),
            },
        },
    },
}


def agex_tool_schemas() -> list[dict]:
    """Return the full set of tools the agent may call."""
    return [
        _PYTHON_SCHEMA,
        _TERMINAL_SCHEMA,
        _WRITE_FILE_SCHEMA,
        _EDIT_FILE_SCHEMA,
    ]
