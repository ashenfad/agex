"""What a type checker reads from a task: pyright checks this file (see
pyproject's ``include``); it is never run."""

from dataclasses import dataclass

from nontainer import Workspace
from typing_extensions import assert_type

from agex import Agent, Outcome, Task

agent = Agent("anthropic:claude-sonnet-5-5")


@dataclass
class Recipe:
    name: str
    minutes: int


@agent.task
def plan_dinner(ingredients: list[str]) -> Recipe:
    """A dinner recipe."""
    ...


@agent.task
async def plan_lunch(ingredients: list[str]) -> Recipe:
    """A lunch recipe."""
    ...


def calls(ws: Workspace) -> None:
    assert_type(plan_dinner, Task[Recipe])
    assert_type(plan_dinner(["eggs"]), Recipe)
    assert_type(plan_dinner(["eggs"], world=ws), Recipe)
    assert_type(plan_dinner.run(["eggs"]), Outcome)
    total: int = plan_dinner(["eggs"]).minutes + 1
    assert_type(total, int)


async def awaits() -> None:
    assert_type(await plan_lunch(["bread"]), Recipe)
    assert_type(await plan_dinner.arun(["eggs"]), Outcome)
