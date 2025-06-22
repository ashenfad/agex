from .helpers import eval_and_get_state


def test_eval_list_literal():
    program = """
x = 3
y = [1, 2, x]
z = []
"""
    state = eval_and_get_state(program)
    assert state.get("y") == [1, 2, 3]
    assert state.get("z") == []


def test_eval_collections():
    """Tests tuples, sets, and dicts."""
    program = """
x = (1, "a", True)
y = {1, "a", True, 1}
z = {"a": 1, "b": x}
w = [1, (2, 3), {"a": {4, 5}}]
"""
    state = eval_and_get_state(program)
    assert state.get("x") == (1, "a", True)
    assert state.get("y") == {1, "a", True}
    assert state.get("z") == {"a": 1, "b": (1, "a", True)}
    assert state.get("w") == [1, (2, 3), {"a": {4, 5}}]


def test_eval_subscript():
    """Tests getting values via subscript."""
    program = """
data = {"a": [10, 20], "b": {"c": 100}}
x = data["a"][1]
y = data["b"]["c"]
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 20
    assert state.get("y") == 100
