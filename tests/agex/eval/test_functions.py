from .helpers import eval_and_get_state


def test_simple_function_def_and_call():
    program = """
def my_func(a, b):
    return a + b

x = my_func(5, 3)
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 8


def test_function_source_text_capture():
    program = """
def my_func(a, b):
    # some comment
    return a + b
"""
    state = eval_and_get_state(program)
    func_obj = state.get("my_func")
    assert func_obj is not None
    assert "def my_func(a, b):" in func_obj.source_text
    assert "return a + b" in func_obj.source_text


def test_lambda_source_text_capture():
    program = "f = lambda x: x * 2"
    state = eval_and_get_state(program)
    func_obj = state.get("f")
    assert func_obj is not None
    assert func_obj.source_text == "lambda x: x * 2"


def test_function_with_no_return():
    program = """
y = 10
def my_func():
    y = 20 # This should be local to the function

my_func()
"""
    state = eval_and_get_state(program)
    # The global y should be unchanged, and the function returns None implicitly.
    assert state.get("y") == 10


def test_recursive_function():
    program = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

x = factorial(5)
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 120


def test_closure_late_binding():
    """
    This is the most important test: it confirms that functions capture their
    enclosing scope by reference (late binding), not by value.
    """
    program = """
def make_adder(x):
    def adder(y):
        return x + y
    return adder

add_5 = make_adder(5)
x = add_5(10) # Should be 15

# Now, we define a function but change the captured variable *before* calling it.
i = 1
def get_i():
    return i

i = 99 # Change the value
y = get_i() # Should return the *new* value
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 15
    assert state.get("y") == 99


def test_lambda_expression():
    program = """
f = lambda a, b: a * b
x = f(3, 4)
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 12


def test_closure_side_effects():
    """Confirms that functions can mutate objects from their closure scope."""
    program = """
l = [1, 2, 3]
d = {"key": "original"}

def mutator():
    l[0] = 99
    d["key"] = "mutated"

mutator()
"""
    state = eval_and_get_state(program)
    assert state.get("l") == [99, 2, 3]
    assert state.get("d")["key"] == "mutated"
