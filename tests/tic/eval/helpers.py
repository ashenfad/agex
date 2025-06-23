from tic.agent import Agent
from tic.eval.core import evaluate_program
from tic.state import Ephemeral, State


def eval_and_get_state(
    program: str, agent: Agent | None = None, state: State | None = None
) -> State:
    """A test helper to evaluate a program and return the final state."""
    agent = agent or Agent()
    state = state or Ephemeral()
    evaluate_program(program, agent, state)
    return state
