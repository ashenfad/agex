import pytest

from tic.eval.error import EvalError

from .helpers import eval_and_get_state


def test_builtin_functions():
    program = """
a = [1, 2, 3]
x = len(a)
y = max(a)
z = min(a)
s = sum(a)
t = str(123)
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3
    assert state.get("y") == 3
    assert state.get("z") == 1
    assert state.get("s") == 6
    assert state.get("t") == "123"


def test_unknown_function_error():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = no_such_function()")
    assert "Function 'no_such_function' is not defined" in str(e.value)


def test_coercion_functions():
    program = """
a = int("10")
b = float("3.14")
c = bool(0)
d = bool(1)
e = bool([])
f = bool([1])
"""
    state = eval_and_get_state(program)
    assert state.get("a") == 10
    assert state.get("b") == 3.14
    assert state.get("c") is False
    assert state.get("d") is True
    assert state.get("e") is False
    assert state.get("f") is True


def test_datastructure_constructors():
    program = """
l = list((1, 2, 3))
t = tuple([1, 2, 3])
s = set([1, 2, 2, 3])
d = dict(a=1, b=2)
empty_l = list()
empty_t = tuple()
empty_s = set()
empty_d = dict()
"""
    state = eval_and_get_state(program)
    assert state.get("l") == [1, 2, 3]
    assert state.get("t") == (1, 2, 3)
    assert state.get("s") == {1, 2, 3}
    assert state.get("d") == {"a": 1, "b": 2}
    assert state.get("empty_l") == []
    assert state.get("empty_t") == ()
    assert state.get("empty_s") == set()
    assert state.get("empty_d") == {}


def test_utility_functions():
    program = """
num = -10.5
abs_num = abs(num)
round_num = round(num)
round_num_2 = round(3.14159, 2)

list1 = [True, True, False]
all_true = all(list1)
any_true = any(list1)

unsorted = [3, 1, 4, 1, 5, 9]
sorted_list = sorted(unsorted)
reversed_list = list(reversed(sorted_list))
zipped = list(zip(sorted_list, reversed_list))
enumerated = list(enumerate(unsorted))
"""
    state = eval_and_get_state(program)
    assert state.get("abs_num") == 10.5
    assert state.get("round_num") == -10
    assert state.get("round_num_2") == 3.14
    assert state.get("all_true") is False
    assert state.get("any_true") is True
    assert state.get("sorted_list") == [1, 1, 3, 4, 5, 9]
    assert state.get("reversed_list") == [9, 5, 4, 3, 1, 1]
    assert state.get("zipped") == [(1, 9), (1, 5), (3, 4), (4, 3), (5, 1), (9, 1)]
    assert state.get("enumerated") == [(0, 3), (1, 1), (2, 4), (3, 1), (4, 5), (5, 9)]


def test_range_function():
    program = """
r1 = range(5)
r2 = range(2, 5)
r3 = range(0, 10, 2)
"""
    state = eval_and_get_state(program)
    assert state.get("r1") == [0, 1, 2, 3, 4]
    assert state.get("r2") == [2, 3, 4]
    assert state.get("r3") == [0, 2, 4, 6, 8]


def test_range_function_error():
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = range(20000)")
    err_msg = str(e.value)
    assert "Error calling builtin function 'range'" in err_msg
    assert "Range exceeds maximum size" in err_msg

    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = range(stop=10)")
    err_msg = str(e.value)
    assert "Error calling builtin function 'range'" in err_msg
    assert "range() does not take keyword arguments" in err_msg


def test_method_calls():
    program = """
my_list = [3, 1, 2]
my_list.append(4)
my_list.sort()

my_dict = {"a": 1}
my_dict.update({"b": 2})
keys = my_dict.keys()

my_str = "  Hello  "
my_str = my_str.strip().upper()
"""
    state = eval_and_get_state(program)
    assert state.get("my_list") == [1, 2, 3, 4]
    assert state.get("my_dict") == {"a": 1, "b": 2}
    # Test that the result was materialized into a list
    assert state.get("keys") == ["a", "b"]
    assert state.get("my_str") == "HELLO"


def test_disallowed_method_call():
    # list.sort is whitelisted, but list.__sizeof__ is not
    with pytest.raises(EvalError) as e:
        eval_and_get_state("x = [].__sizeof__()")
    assert "Method '__sizeof__' is not allowed on type 'list'" in str(e.value)


def test_user_function_is_python_callable():
    """
    Tests that a UserFunction retrieved from state can be called from Python.
    """
    program = """
def add(a, b):
    return a + b
"""
    state = eval_and_get_state(program)
    add_func = state.get("add")

    assert add_func is not None
    # Call the function directly from Python
    result = add_func(10, 20)
    assert result == 30

    # Test with keyword arguments
    result_kwargs = add_func(a=5, b=6)
    assert result_kwargs == 11
