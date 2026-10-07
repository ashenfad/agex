"""Types for the task tests that run on every rung, in a module of their
own: a worker process imports them by name, and a dud guest without
this module rebuilds it from its source, which needs nothing past the
standard library."""

from dataclasses import dataclass


@dataclass
class Score:
    student: str
    total: int


@dataclass
class Ranking:
    best: str
    scores: list[Score]


class Directory:
    """A live resource a task is handed as a capability."""

    def __init__(self) -> None:
        self.looked_up: list[str] = []

    def lookup(self, key: str) -> str:
        self.looked_up.append(key)
        return key.upper()
