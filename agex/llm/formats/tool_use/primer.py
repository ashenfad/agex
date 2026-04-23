"""System-prompt addendum for the tool-use wire format.

Schema-level docs carry most of the instruction (tool descriptions,
parameter descriptions).  This primer adds the agex-specific semantics
that can't be expressed in a JSON Schema.
"""

TOOL_USE_FORMAT_PRIMER = """
You drive agex tasks by calling tools, not by writing XML tags.

Per turn you may call any combination of python_action, terminal_action,
write_file, and edit_file — they execute in the order they appear.
Python emissions share state: later python_action calls see variables
assigned by earlier ones.  File tools run before subsequent python_action
calls, so you can write a helper module and import it in the same turn.

Inside python_action's `code`:
- task_success(value) finishes the task with `value` as the result.
- task_fail(msg) ends the task with an error message.
- task_clarify(prompt) asks your caller a question.
- None of the above: your turn ends normally and you'll see any
  printed / view_image() output at the start of the next turn.

write_file places Python modules under /helpers.  Use mode="append" to
extend an existing file.

edit_file must match `search` exactly (including whitespace).  Prefer
insert_after / insert_before over a replace that repeats the search text
— the latter duplicates on accidental re-runs.

The first tool_result (stdout, confirmation) appears in subsequent
messages.  Treat it as data from your own execution, not as a message
from the user.
"""
