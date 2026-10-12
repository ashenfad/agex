"""A task: a typed function an agent answers.

    AGEX_MODEL=anthropic:claude-sonnet-5-5 python examples/task.py

The docstring is the job and the body stays empty. The agent works on a
scratch world of its own and hands back its answer with
``task.success(...)``; the caller gets a ``Recipe``, checked against
the annotation, or a ``TaskFailed``.
"""

import os
from dataclasses import dataclass

from agex import Agent, TaskFailed


@dataclass
class Recipe:
    name: str
    steps: list[str]
    minutes: int


def main(agent: Agent) -> None:
    @agent.task
    def plan_dinner(ingredients: list[str]) -> Recipe:
        """A dinner recipe that uses only these ingredients, plus salt and
        oil. ``minutes`` is the total time, preparation included."""
        ...

    try:
        recipe = plan_dinner(["eggs", "spinach", "feta"])
    except TaskFailed as error:
        print(f"no recipe: {error}")
        return
    print(f"{recipe.name} ({recipe.minutes} min)")
    for n, step in enumerate(recipe.steps, 1):
        print(f"  {n}. {step}")


if __name__ == "__main__":
    main(Agent(os.environ.get("AGEX_MODEL", "anthropic:claude-sonnet-5-5")))
