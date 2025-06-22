from .helpers import eval_and_get_state


def test_list_slicing_full():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[:]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [1, 2, 3, 4, 5]
    assert id(state.get("sliced")) != id(state.get("items"))


def test_list_slicing_start():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[2:]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [3, 4, 5]


def test_list_slicing_end():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[:3]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [1, 2, 3]


def test_list_slicing_middle():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[1:4]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [2, 3, 4]


def test_list_slicing_with_step():
    program = """
items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sliced = items[1:8:2]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [2, 4, 6, 8]


def test_list_slicing_negative_start():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[-3:]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [3, 4, 5]


def test_list_slicing_negative_end():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[:-2]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [1, 2, 3]


def test_list_slicing_negative_step():
    program = """
items = [1, 2, 3, 4, 5]
sliced = items[::-1]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == [5, 4, 3, 2, 1]


def test_string_slicing():
    program = """
text = "hello world"
sliced = text[6:]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == "world"


def test_string_slicing_reverse():
    program = """
text = "hello"
sliced = text[::-1]
"""
    state = eval_and_get_state(program)
    assert state.get("sliced") == "olleh"
