from .helpers import eval_and_get_state


def test_eval_assignment():
    state = eval_and_get_state("x = 1\ny = x")
    assert state.get("x") == 1
    assert state.get("y") == 1
