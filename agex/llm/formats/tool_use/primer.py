"""System-prompt addendum for the tool-use wire format.

Schema-level docs carry most of the instruction (tool descriptions,
parameter descriptions).  This primer adds the agex-specific semantics
that can't be expressed in a JSON Schema.
"""

_PRIMER_BASE = """
Tools are the entire interface: every turn must call at least one of
python_action, terminal_action, write_file, or edit_file.  Plain
assistant text is not a reply channel — it doesn't execute anything,
doesn't finish the task, and the provider is configured to require a
tool call.  If you have a question, call ``task_clarify`` inside
python_action; if you want to report, do it via ``print(...)`` in
python_action so the output shows up in your next tool_result.

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
extend an existing file.  Create the file before importing from it —
don't assume a module or function exists unless you wrote it (or saw
it in an earlier tool_result).

edit_file does one thing: swap `search` for `replace`.  `search` must
match the file exactly (whitespace is significant) and only work on
text you have already observed — either you wrote it this session or
it appeared in a tool_result (e.g. from `cat`).  If you want to add
new content to a file that already exists, prefer write_file with
mode="append" — append can't miss a search target that was never
there.  To insert new content around an existing anchor, include the
anchor itself in `replace` (search for `def foo():` and replace with
`def foo():\n<new line>` to add a line underneath it).

The first tool_result (stdout, confirmation) appears in subsequent
messages.  Treat it as data from your own execution, not as a message
from the user.
"""

_NATIVE_THINKING_ADDENDUM = """
Your provider delivers reasoning as native thinking blocks; use them
for step-by-step reasoning.  Tool calls are still the only way to
advance or finish the task.
"""


def format_primer(native_thinking: bool = False) -> str:
    """Return the tool-use primer text, with a short addendum appended
    when ``native_thinking=True`` so the model knows to use its native
    thinking/text channels instead of narration-in-schema.
    """
    if native_thinking:
        return _PRIMER_BASE + _NATIVE_THINKING_ADDENDUM
    return _PRIMER_BASE
