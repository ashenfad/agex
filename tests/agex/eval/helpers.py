from agex.agent import Agent
from agex.eval.core import evaluate_program
from agex.state import Ephemeral, State


def eval_and_get_state(
    program: str,
    agent: Agent | None = None,
    state: State | None = None,
    include_numpy: bool = False,
) -> State:
    """A test helper to evaluate a program and return the final state."""
    agent = agent or Agent()
    if include_numpy:
        import numpy as np

        agent.module(np)
    if state is None:
        state = Ephemeral()
    evaluate_program(program, agent, state)
    return state
