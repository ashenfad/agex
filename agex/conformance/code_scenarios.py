"""The shape corpus, for tasks agent code defines (``@agex.task``).

The agent's own code defines a task and calls it, and a helper agent
runs each call. Each harness writes the calling code in its own
language, from the task as JSON Schemas:

- **Values:** the calling code gets the value built as its own classes,
  records nested in records included, and tables, arrays and bytes come
  back as declared.
- **Many at once:** ``.map`` returns the values in the order of its
  items.
- **Endings:** a helper that fails raises the failure in the calling
  code, and one that asks raises its question.
- **Refusals:** a task that can't be one is refused where it is
  defined: no docstring, a parameter with no annotation, a live type,
  or a parameter named like what the helper's world binds.

What a harness does around this (helpers as delegate branches, helper
time off the script's clock, one commit for the calling script) is the
harness's own, and its tests pin it.
"""

from __future__ import annotations

from .scenarios import (
    ARRAY,
    BYTES,
    DIRECTORY,
    RANKED,
    RANKING,
    SCORE,
    SCORES,
    STR,
    TABLE,
    list_of,
    ref,
)
from .shape import (
    CodeScenario,
    Input,
    Param,
    TaskDef,
    code_asked,
    code_call,
    code_failed,
    code_map,
    code_refused,
    got,
    got_each,
    task_fail,
    task_needs_input,
    task_success,
)

__all__ = ["CODE_SCENARIOS"]

RANK = TaskDef(
    name="rank",
    instructions="Rank the scores, best first.",
    params=(Param(name="scores", type=list_of(ref("Score"))),),
    returns=ref("Ranking"),
    defs={"Score": SCORE, "Ranking": RANKING},
)

SHOUT = TaskDef(
    name="shout",
    instructions="Shout the word back.",
    params=(Param(name="word", type=STR),),
    returns=STR,
)


def made(name: str, tp: dict, **defs: dict) -> TaskDef:
    """A task that makes a value of type ``tp`` from nothing much."""
    return TaskDef(
        name=name,
        instructions=f"Make a {name}.",
        params=(Param(name="label", type=STR),),
        returns=tp,
        defs=defs,
    )


OWN_RECORDS = CodeScenario(
    name="code-gets-its-own-records-back",
    summary=(
        "Agent code defines a task over records of its own; the helper "
        "builds the value of the task's types, and the calling code gets "
        "it built as its own classes, nested records included."
    ),
    task=RANK,
    acts=(code_call(task_success(RANKED), scores=SCORES),),
    expect=(got(RANKED),),
)

TABLES_AND_ARRAYS = CodeScenario(
    name="code-gets-tables-arrays-and-bytes-back",
    summary="Tables, arrays and bytes come back as declared.",
    task=made(
        "bundle",
        ref("Bundle"),
        Bundle={
            "type": "object",
            "description": "A table, an array and some bytes.",
            "properties": {"table": TABLE, "grid": ARRAY, "blob": BYTES},
            "required": ["table", "grid", "blob"],
            "additionalProperties": False,
        },
    ),
    needs=("tables", "arrays"),
    acts=(
        code_call(
            task_success(
                {"table": {"x": [1, 2]}, "grid": [[1, 2], [3, 4]], "blob": "aGk="}
            ),
            label="b",
        ),
    ),
    expect=(got({"table": {"x": [1, 2]}, "grid": [[1, 2], [3, 4]], "blob": "aGk="}),),
)

MAPPED = CodeScenario(
    name="code-maps-a-task-in-order",
    summary="A task mapped over items returns each item's value, in order.",
    task=SHOUT,
    acts=(
        code_map(
            [{"word": "a"}, {"word": "b"}, {"word": "c"}],
            task_success(Input(name="word")),
        ),
    ),
    expect=(got_each("a", "b", "c"),),
)

FAILED = CodeScenario(
    name="code-sees-a-helper-fail",
    summary="A helper that fails raises its reason in the calling code.",
    task=SHOUT,
    acts=(code_call(task_fail("no voice today"), word="hi"),),
    expect=(code_failed("no voice today"),),
)

ASKED = CodeScenario(
    name="code-sees-a-helper-ask",
    summary=(
        "A helper that asks raises its question in the calling code, which "
        "has no model to answer it."
    ),
    task=SHOUT,
    acts=(code_call(task_needs_input("Shout how loud?"), word="hi"),),
    expect=(code_asked("Shout how loud?"),),
)


def refused(name: str, summary: str, task: TaskDef, *names: str) -> CodeScenario:
    return CodeScenario(
        name=name,
        summary=summary,
        task=task,
        acts=(code_call(word="hi"),),
        expect=(code_refused(*names),),
    )


NO_DOCSTRING = refused(
    "code-is-refused-a-task-without-a-docstring",
    "A task with no docstring has no job to give a helper, so it is refused.",
    TaskDef(name="shout", instructions="", params=SHOUT.params, returns=STR),
    "shout",
    "docstring",
)

NO_ANNOTATION = refused(
    "code-is-refused-a-parameter-without-a-type",
    "A parameter with no annotation is refused, naming the parameter.",
    TaskDef(
        name="shout",
        instructions=SHOUT.instructions,
        params=(Param(name="word", type=None),),
        returns=STR,
    ),
    "word",
    "annotation",
)

LIVE_TYPE = refused(
    "code-is-refused-a-live-type",
    "A task takes and returns values: a live type is refused, on every rung.",
    TaskDef(
        name="look",
        instructions="Look the word up.",
        params=(Param(name="word", type=STR),),
        returns=ref("Directory"),
        defs={"Directory": DIRECTORY},
    ),
    "look",
    "live",
)

RESERVED = refused(
    "code-is-refused-a-parameter-the-helper-binds",
    "A parameter named like what the helper's world binds (task) is refused.",
    TaskDef(
        name="shout",
        instructions=SHOUT.instructions,
        params=(Param(name="task", type=STR),),
        returns=STR,
    ),
    "task",
)

CODE_SCENARIOS: list[CodeScenario] = [
    OWN_RECORDS,
    TABLES_AND_ARRAYS,
    MAPPED,
    FAILED,
    ASKED,
    NO_DOCSTRING,
    NO_ANNOTATION,
    LIVE_TYPE,
    RESERVED,
]
