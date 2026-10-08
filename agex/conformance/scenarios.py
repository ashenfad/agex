"""The shape corpus: agex's task API, as scenarios every harness runs.

- **Typed values:** a task hands back a value of its return type, built
  with the record classes bound in its world; the same value comes back
  on every rung, for data, tables, arrays and bytes; a value that
  doesn't fit is refused at ``task.success``, and the model tries again.
- **Endings:** a task fails with its reason, and a provider error
  interrupts it.
- **Asking:** a task asks with ``task.needs_input``, keeps its world,
  and carries on from the answer, through its outcome or by its ref
  after a restart; it can ask again; a resume missing a live input is
  refused by name.
- **Live inputs:** a live input is a capability whose methods agent
  code calls on the host; what can't leave the process (a live return
  type, an input mixing data with a live part) is refused off it before
  the model is asked anything.
- **Keeping:** ``keep`` keeps the fork, with the task's value in its
  plane.
"""

from __future__ import annotations

from typing import Any

from nontainer.conformance.corpus import fails, says

from .shape import (
    Input,
    LiveCall,
    MethodCall,
    OutcomeExp,
    Param,
    PlaneExp,
    TaskDef,
    TaskScenario,
    call,
    failure,
    question,
    refusal,
    restart,
    resume,
    success,
    task_fail,
    task_needs_input,
    task_success,
)

__all__ = ["SCENARIOS"]


# -- types ------------------------------------------------------------------------------


def ref(name: str) -> dict[str, Any]:
    return {"$ref": f"#/$defs/{name}"}


def list_of(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def dict_of(values: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "additionalProperties": values}


def record(description: str, **fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


STR = {"type": "string"}
INT = {"type": "integer"}
NONE = {"type": "null"}
TABLE = {"x-kind": "table"}
ARRAY = {"x-kind": "array"}
BYTES = {"x-kind": "bytes"}

SCORE = record("One student's total.", student=STR, total=INT)
RANKING = record(
    "The best student, and the scores ranked.", best=STR, scores=list_of(ref("Score"))
)
DIRECTORY = {
    "x-kind": "live",
    "description": "The school's directory, kept on the host.",
    "methods": {
        "lookup": {
            "description": "A student's name as the directory files it.",
            "params": [{"name": "key", "type": STR}],
            "returns": STR,
            "replies": [
                {"args": ["ada"], "returns": "ADA"},
                {"args": ["bo"], "returns": "BO"},
            ],
        }
    },
}
SCORES = [{"student": "ada", "total": 7}, {"student": "bo", "total": 2}]
RANKED = {"best": "ada", "scores": SCORES}

RANK = TaskDef(
    name="rank",
    instructions="Rank the scores, best first.",
    params=(Param(name="scores", type=list_of(ref("Score"))),),
    returns=ref("Ranking"),
    defs={"Score": SCORE, "Ranking": RANKING},
)
LOOK = TaskDef(
    name="look",
    instructions="Look up the best student in the directory.",
    params=(
        Param(name="scores", type=list_of(ref("Score"))),
        Param(name="directory", type=ref("Directory")),
    ),
    returns=STR,
    defs={"Score": SCORE, "Directory": DIRECTORY},
)
QUESTION = "Rank by total, or by the best single score?"
LOOKED_UP = (LiveCall(input="directory", method="lookup", args=("ada",)),)


def echo(name: str, tp: dict[str, Any], **defs: Any) -> TaskDef:
    """A task that hands back its input ``name``, of type ``tp``."""
    return TaskDef(
        name="echo",
        instructions=f"Hand back the input {name!r} as it is.",
        params=(Param(name=name, type=tp),),
        returns=tp,
        defs=defs,
    )


# -- typed values -----------------------------------------------------------------------

TYPED_VALUE = TaskScenario(
    name="a-task-hands-back-a-typed-value",
    summary=(
        "Agent code builds the value with the record classes bound in its "
        "world and passes it to task.success; the call returns it, of the "
        "return type, and keeps no world."
    ),
    task=RANK,
    acts=(call(task_success(RANKED), scores=SCORES),),
    expect=(success(RANKED, asked=1, kept=False),),
)

DATA_ROUND_TRIP = TaskScenario(
    name="data-comes-back-the-same",
    summary=(
        "Records, lists, dicts and optional values handed to a task come "
        "back from it the same, on every rung."
    ),
    task=echo(
        "scores",
        dict_of({"anyOf": [list_of(ref("Score")), NONE]}),
        Score=SCORE,
    ),
    acts=(
        call(
            task_success(Input(name="scores")),
            scores={"math": SCORES, "art": [], "music": None},
        ),
    ),
    expect=(success({"math": SCORES, "art": [], "music": None}, asked=1),),
)

TABLE_ROUND_TRIP = TaskScenario(
    name="a-table-comes-back-the-same",
    summary="A table handed to a task comes back from it the same, on every rung.",
    task=echo("frame", TABLE),
    needs=("tables",),
    acts=(
        call(
            task_success(Input(name="frame")),
            frame={"name": ["ada", "bo"], "score": [7, 2]},
        ),
    ),
    expect=(success({"name": ["ada", "bo"], "score": [7, 2]}, asked=1),),
)

ARRAY_ROUND_TRIP = TaskScenario(
    name="an-array-comes-back-the-same",
    summary="An array handed to a task comes back from it the same, on every rung.",
    task=echo("values", ARRAY),
    needs=("arrays",),
    acts=(call(task_success(Input(name="values")), values=[[1, 2], [3, 4]]),),
    expect=(success([[1, 2], [3, 4]], asked=1),),
)

BYTES_ROUND_TRIP = TaskScenario(
    name="bytes-come-back-the-same",
    summary="Bytes handed to a task come back from it the same, on every rung.",
    task=echo("data", BYTES),
    acts=(call(task_success(Input(name="data")), data="AAH/"),),
    expect=(success("AAH/", asked=1),),
)

NOT_FITTING = TaskScenario(
    name="a-value-that-does-not-fit-is-refused-at-task-success",
    summary=(
        "task.success refuses a value that isn't of the return type, as an "
        "error in agent code; the model tries again, and the call returns "
        "the value that fits."
    ),
    task=RANK,
    acts=(
        call(
            task_success({"best": "ada", "scores": [{"student": "ada", "total": "7"}]}),
            task_success(RANKED),
            scores=SCORES,
        ),
    ),
    expect=(success(RANKED, asked=2),),
)

# -- endings ----------------------------------------------------------------------------

FAILS = TaskScenario(
    name="a-task-fails-with-its-reason",
    summary="task.fail ends the task failed, with the reason agent code gave.",
    task=RANK,
    acts=(call(task_fail("there are no scores to rank"), scores=[]),),
    expect=(failure("there are no scores to rank", asked=1, kept=False),),
)

INTERRUPTED = TaskScenario(
    name="a-provider-error-interrupts-the-task",
    summary="A provider error the loop can't get past ends the task interrupted.",
    task=RANK,
    acts=(call(fails(), scores=SCORES),),
    expect=(OutcomeExp(status="interrupted", asked=1),),
)

# -- asking -----------------------------------------------------------------------------

ASKS = TaskScenario(
    name="a-task-asks-and-carries-on-from-the-answer",
    summary=(
        "task.needs_input ends the task needs_input with its question, "
        "keeping its world with the inputs stored; a resume through the "
        "outcome carries on there, and a resume run to its end keeps nothing."
    ),
    task=RANK,
    acts=(
        call(task_needs_input(QUESTION), scores=SCORES),
        resume("By total.", task_success(RANKED)),
    ),
    expect=(
        question(
            QUESTION, asked=1, plane=PlaneExp(status="needs_input", stored=("scores",))
        ),
        success(RANKED, asked=1, kept=False),
    ),
)

ASKS_AGAIN = TaskScenario(
    name="a-task-can-ask-again",
    summary="A resumed task can ask again, and a second answer carries it on.",
    task=RANK,
    acts=(
        call(task_needs_input(QUESTION), scores=SCORES),
        resume("By total.", task_needs_input("Ties first or last?")),
        resume("Last.", task_success(RANKED)),
    ),
    expect=(
        question(QUESTION, asked=1),
        question("Ties first or last?", asked=1),
        success(RANKED, asked=1, kept=False),
    ),
)

RESUME_AFTER_RESTART = TaskScenario(
    name="a-task-resumes-by-ref-after-a-restart",
    summary=(
        "A task that asks on a fork of a world keeps the fork; after a "
        "restart, a resume by its ref through that world carries on with "
        "the stored inputs and the live input passed again."
    ),
    task=LOOK,
    acts=(
        call(task_needs_input(QUESTION), world=True, scores=SCORES, directory={}),
        restart(),
        resume(
            "By total.",
            task_success(MethodCall(input="directory", method="lookup", args=("ada",))),
            by="ref",
            live=("directory",),
        ),
    ),
    expect=(
        question(
            QUESTION, asked=1, plane=PlaneExp(status="needs_input", stored=("scores",))
        ),
        success("ADA", asked=1, kept=False, calls=LOOKED_UP),
    ),
)

RESUME_MISSING_LIVE = TaskScenario(
    name="a-resume-missing-a-live-input-is-refused-by-name",
    summary=(
        "A live input isn't stored with the task, so a resume by ref must "
        "pass it again: one that doesn't is refused, naming it, before the "
        "model is asked anything, and the task still waits for its answer."
    ),
    task=LOOK,
    acts=(
        call(task_needs_input(QUESTION), world=True, scores=SCORES, directory={}),
        restart(),
        resume("By total.", by="ref"),
        resume(
            "By total.",
            task_success(MethodCall(input="directory", method="lookup", args=("ada",))),
            by="ref",
            live=("directory",),
        ),
    ),
    expect=(
        question(QUESTION, asked=1),
        refusal("directory"),
        success("ADA", asked=1, calls=LOOKED_UP),
    ),
)

# -- live inputs ------------------------------------------------------------------------

CAPABILITY = TaskScenario(
    name="a-live-input-is-a-capability",
    summary=(
        "A live input is handed over as it is: agent code calls its methods, "
        "which run on the host, on every rung."
    ),
    task=LOOK,
    acts=(
        call(
            task_success(MethodCall(input="directory", method="lookup", args=("ada",))),
            scores=SCORES,
            directory={},
        ),
    ),
    expect=(success("ADA", asked=1, calls=LOOKED_UP),),
)

LIVE_RETURN = TaskScenario(
    name="a-live-return-type-is-refused-off-in-process",
    summary=(
        "Off the process, a task whose return type is live is refused before "
        "the model is asked anything: its value couldn't come back."
    ),
    task=TaskDef(
        name="directory",
        instructions="Make a directory.",
        returns=ref("Directory"),
        defs={"Directory": DIRECTORY},
    ),
    where="off-in-process",
    acts=(call(says("never reached")),),
    expect=(refusal("Directory"),),
)

MIXED_INPUT = TaskScenario(
    name="an-input-mixing-data-and-a-live-part-is-refused-off-in-process",
    summary=(
        "Off the process, an input that holds live objects inside data is "
        "refused before the model is asked anything, naming the input."
    ),
    task=TaskDef(
        name="look_all",
        instructions="Look up ada in each directory.",
        params=(Param(name="directories", type=dict_of(ref("Directory"))),),
        returns=STR,
        defs={"Directory": DIRECTORY},
    ),
    where="off-in-process",
    acts=(call(says("never reached"), directories={"main": {}}),),
    expect=(refusal("directories"),),
)

# -- keeping ----------------------------------------------------------------------------

KEEPS = TaskScenario(
    name="keep-keeps-the-fork-with-its-value",
    summary=(
        "keep keeps a task's fork when it is done; its plane holds the "
        "inputs and the value, and says the task succeeded."
    ),
    task=RANK,
    acts=(call(task_success(RANKED), world=True, keep=True, scores=SCORES),),
    expect=(
        success(
            RANKED,
            asked=1,
            kept=True,
            plane=PlaneExp(status="success", stored=("scores",), value=True),
        ),
    ),
)

SCENARIOS: tuple[TaskScenario, ...] = (
    TYPED_VALUE,
    DATA_ROUND_TRIP,
    TABLE_ROUND_TRIP,
    ARRAY_ROUND_TRIP,
    BYTES_ROUND_TRIP,
    NOT_FITTING,
    FAILS,
    INTERRUPTED,
    ASKS,
    ASKS_AGAIN,
    RESUME_AFTER_RESTART,
    RESUME_MISSING_LIVE,
    CAPABILITY,
    LIVE_RETURN,
    MIXED_INPUT,
    KEEPS,
)
"""Every shape scenario, in the order above."""
