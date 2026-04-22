"""System-prompt addendum for the tool-use wire format.

Schema-level docs carry most of the instruction (tool descriptions,
parameter descriptions).  This primer adds the agex-specific semantics
that can't be expressed in a JSON Schema.
"""

TOOL_USE_FORMAT_PRIMER = """
You drive agex tasks by calling tools, not by writing XML tags.

Per turn, call EITHER python_action OR terminal_action (not both) for
the main action.  You may also call write_file and/or edit_file
alongside it to prepare files before the main action runs.

Inside python_action's `code`:
- task_success(value) finishes the task with `value` as the result.
- task_fail(msg) ends the task with an error message.
- task_clarify(prompt) asks your caller a question.
- task_continue() runs the code and advances to the next turn so you
  can inspect output.  Default if you omit an explicit task_* call.

Terminal actions implicitly task_continue() — use python_action when
you're ready to finish.

write_file places Python modules under /helpers.  Use mode="append" to
extend an existing file.

edit_file must match `search` exactly (including whitespace).  Prefer
insert_after/insert_before over a replace that repeats the search text
— the latter duplicates on accidental re-runs.

Output (stdout, tool_result) appears in subsequent tool_result content.
Treat it as data from your own execution, not a message from the user.
"""
