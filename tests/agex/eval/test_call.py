import pytest

from agex.eval.user_errors import AgexAttributeError, AgexError, AgexNameError

from .helpers import eval_and_get_state


def test_builtin_functions():
    program = """
a = [1, 2, 3]
x = len(a)
y = max(a)
z = min(a)
s = sum(a)
t = str(123)
p = pow(2, 3)
b = bytes([65, 66, 67])
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3
    assert state.get("y") == 3
    assert state.get("z") == 1
    assert state.get("s") == 6
    assert state.get("t") == "123"
    assert state.get("p") == 8
    assert state.get("b") == b"ABC"


def test_unknown_function_error():
    with pytest.raises(AgexNameError) as e:
        eval_and_get_state("x = no_such_function()")
    assert "name 'no_such_function' is not defined" in str(e.value)


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
r1 = list(range(5))
r2 = list(range(2, 5))
r3 = list(range(0, 10, 2))
"""
    state = eval_and_get_state(program)
    assert state.get("r1") == [0, 1, 2, 3, 4]
    assert state.get("r2") == [2, 3, 4]
    assert state.get("r3") == [0, 2, 4, 6, 8]


def test_range_assignment():
    # Test that ranges can be assigned directly now (they're pickleable)
    program = "x = range(20000)"
    state = eval_and_get_state(program)
    r = state.get("x")
    assert isinstance(r, range)
    assert len(r) == 20000
    assert list(r)[:5] == [0, 1, 2, 3, 4]  # Check first few elements


def test_method_calls():
    program = """
my_list = [3, 1, 2]
my_list.append(4)
my_list.sort()

my_dict = {"a": 1}
my_dict.update({"b": 2})
keys = list(my_dict.keys())

my_str = "  Hello  "
my_str = my_str.strip().upper()
"""
    state = eval_and_get_state(program)
    assert state.get("my_list") == [1, 2, 3, 4]
    assert state.get("my_dict") == {"a": 1, "b": 2}
    # Test that the result was materialized into a list
    assert list(state.get("keys")) == ["a", "b"]
    assert state.get("my_str") == "HELLO"


def test_disallowed_method_call():
    # list.sort is whitelisted, but list.__sizeof__ is not
    with pytest.raises(AgexAttributeError) as e:
        eval_and_get_state("x = [].__sizeof__()")
    assert "object has no attribute '__sizeof__'" in str(e.value)


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


def test_sort_with_key_function():
    """
    Tests that `sorted()` and `list.sort()` work with a tic UserFunction as a key.
    """
    program = """
data = [("a", 3), ("b", 1), ("c", 2)]

def get_second_element(t):
    return t[1]

# Test the sorted() builtin
sorted_data = sorted(data, key=get_second_element)

# Test the list.sort() method
data_to_sort = [("x", 9), ("y", 8), ("z", 7)]
data_to_sort.sort(key=get_second_element)
"""
    state = eval_and_get_state(program)

    # Check the result of the sorted() builtin
    expected_sorted = [("b", 1), ("c", 2), ("a", 3)]
    assert state.get("sorted_data") == expected_sorted

    # Check the result of the list.sort() method
    expected_in_place = [("z", 7), ("y", 8), ("x", 9)]
    assert state.get("data_to_sort") == expected_in_place


def test_sort_with_lambda_key():
    """
    Tests that `sorted()` and `list.sort()` work with a tic lambda as a key.
    """
    program = """
data = [("a", 3), ("b", 1), ("c", 2)]

# Test the sorted() builtin with a lambda
sorted_data = sorted(data, key=lambda t: t[1])

# Test the list.sort() method with a lambda
data_to_sort = [("x", 9), ("y", 8), ("z", 7)]
data_to_sort.sort(key=lambda t: t[1])
"""
    state = eval_and_get_state(program)

    # Check the result of the sorted() builtin
    expected_sorted = [("b", 1), ("c", 2), ("a", 3)]
    assert state.get("sorted_data") == expected_sorted

    # Check the result of the list.sort() method
    expected_in_place = [("z", 7), ("y", 8), ("x", 9)]
    assert state.get("data_to_sort") == expected_in_place


def test_map_and_filter_builtins():
    """Tests that map() and filter() work with user-defined functions."""
    program = """
numbers = [1, 2, 3, 4, 5]

def square(x):
    return x * x

# Test map with a named function
squared_numbers = list(map(square, numbers))

# Test filter with a lambda
even_numbers = list(filter(lambda x: x % 2 == 0, numbers))

# Test map and filter together
even_squares = list(map(square, filter(lambda x: x % 2 == 0, numbers)))
"""
    state = eval_and_get_state(program)
    assert state.get("squared_numbers") == [1, 4, 9, 16, 25]
    assert state.get("even_numbers") == [2, 4]
    assert state.get("even_squares") == [4, 16]


def test_type_introspection_builtins():
    """Tests the isinstance() and type() builtins."""
    program = """
a = 1
b = "hello"
c = [1, 2]
d = {"key": "value"}

is_a_int = isinstance(a, int)
is_a_str = isinstance(a, str)
is_b_str = isinstance(b, str)
is_c_list = isinstance(c, list)
is_d_dict = isinstance(d, dict)

type_a = type(a)
type_b = type(b)
type_c = type(c)
type_d = type(d)

def my_func():
    pass

is_f_func = isinstance(my_func, type(my_func))
"""
    state = eval_and_get_state(program)

    assert state.get("is_a_int") is True
    assert state.get("is_a_str") is False
    assert state.get("is_b_str") is True
    assert state.get("is_c_list") is True
    assert state.get("is_d_dict") is True

    assert state.get("type_a").__name__ == "int"
    assert state.get("type_b").__name__ == "str"
    assert state.get("type_c").__name__ == "list"
    assert state.get("type_d").__name__ == "dict"

    # This is a bit of a trick to test the type of a user function,
    # since we don't have a direct name for UserFunction in the environment.
    assert state.get("is_f_func") is True


def test_isinstance_with_tuple():
    """Tests isinstance() with tuple of types."""
    program = """
# Test with native types
result1 = isinstance(5, (int, str))      # Should be True (5 is int)
result2 = isinstance("hello", (int, str))  # Should be True ("hello" is str)
result3 = isinstance(5.5, (int, str))    # Should be False (5.5 is float)
result4 = isinstance([], (list, dict))   # Should be True ([] is list)
result5 = isinstance({}, (list, dict))   # Should be True ({} is dict)
result6 = isinstance(5, (str, dict))     # Should be False (5 is not str or dict)

# Test with user-defined classes
class MyClass:
    def __init__(self, value):
        self.value = value

from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float

# Create instances
obj = MyClass(42)
p = Point(1.0, 2.0)

# Test isinstance with tuples on custom types
result7 = isinstance(p, (Point, MyClass))      # Should be True
result8 = isinstance(obj, (Point, MyClass))    # Should be True  
result9 = isinstance(p, (MyClass, str))        # Should be False
result10 = isinstance(obj, (Point, str))       # Should be False
result11 = isinstance(5, (Point, MyClass))     # Should be False

# Test edge cases
result12 = isinstance(5, ())               # Empty tuple - should be False
result13 = isinstance(5, (int,))           # Single-element tuple - should be True
result14 = isinstance(5, (str,))           # Single-element tuple - should be False

# Test mixed native and custom types
result15 = isinstance("hello", (str, Point))    # Should be True
result16 = isinstance(p, (int, Point))         # Should be True
"""
    state = eval_and_get_state(program)

    # Test native types
    assert state.get("result1") is True
    assert state.get("result2") is True
    assert state.get("result3") is False
    assert state.get("result4") is True
    assert state.get("result5") is True
    assert state.get("result6") is False

    # Test custom types
    assert state.get("result7") is True
    assert state.get("result8") is True
    assert state.get("result9") is False
    assert state.get("result10") is False
    assert state.get("result11") is False

    # Test edge cases
    assert state.get("result12") is False
    assert state.get("result13") is True
    assert state.get("result14") is False

    # Test mixed types
    assert state.get("result15") is True
    assert state.get("result16") is True


def test_call_on_non_callable():
    with pytest.raises(AgexError) as e:
        eval_and_get_state("x = 123()")
    assert "'object' is not callable" in str(e.value)
