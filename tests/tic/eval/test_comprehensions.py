from .helpers import eval_and_get_state


def test_list_comprehension():
    program = """
numbers = [1, 2, 3, 4, 5]
squares = [n * n for n in numbers]
"""
    state = eval_and_get_state(program)
    assert state.get("squares") == [1, 4, 9, 16, 25]


def test_list_comprehension_with_if():
    program = """
numbers = [1, 2, 3, 4, 5]
even_squares = [n * n for n in numbers if n % 2 == 0]
"""
    state = eval_and_get_state(program)
    assert state.get("even_squares") == [4, 16]


def test_set_comprehension():
    program = """
words = ["apple", "banana", "apple", "cherry"]
unique_lengths = {len(w) for w in words}
"""
    state = eval_and_get_state(program)
    assert state.get("unique_lengths") == {5, 6}


def test_dict_comprehension():
    program = """
words = ["apple", "banana"]
word_lengths = {w: len(w) for w in words}
"""
    state = eval_and_get_state(program)
    assert state.get("word_lengths") == {"apple": 5, "banana": 6}


def test_nested_comprehension():
    program = """
matrix = [[1, 2], [3, 4]]
flat = [item for row in matrix for item in row]
"""
    state = eval_and_get_state(program)
    assert state.get("flat") == [1, 2, 3, 4]


def test_generator_expression_materialized():
    program = """
numbers = (n for n in [1, 2, 3])
"""
    state = eval_and_get_state(program)
    assert state.get("numbers") == [1, 2, 3]


def test_comprehension_scope_does_not_leak():
    program = """
n = 100
numbers = [1, 2, 3, 4, 5]
squares = [n * n for n in numbers]
"""
    state = eval_and_get_state(program)
    assert state.get("squares") == [1, 4, 9, 16, 25]
    # The crucial test: 'n' from the comprehension should not have leaked.
    assert state.get("n") == 100


def test_comprehension_with_destructuring():
    program = """
d = {"a": 1, "b": 2}
l = [k + str(v) for k, v in d.items()]
l.sort()
"""
    state = eval_and_get_state(program)
    assert state.get("l") == ["a1", "b2"]
