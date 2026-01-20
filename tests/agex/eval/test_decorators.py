import functools

from agex.agent import Agent

from .helpers import eval_and_get_state


def test_identity_decorator():
    program = """
def identity(f):
    return f

@identity
def foo():
    return 1

x = foo()
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 1


def test_wrapper_decorator():
    program = """
def double_result(f):
    def wrapper(*args):
        return f(*args) * 2
    return wrapper

@double_result
def foo():
    return 10

x = foo()
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 20


def test_stacked_decorators():
    program = """
def add_one(f):
    def wrapper(*args):
        return f(*args) + 1
    return wrapper

def double_result(f):
    def wrapper(*args):
        return f(*args) * 2
    return wrapper

@double_result
@add_one
def foo():
    return 10

# logic: (10 + 1) * 2 = 22
x = foo()
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 22


def test_decorator_with_arguments():
    program = """
def add_n(n):
    def decorator(f):
        def wrapper(*args):
            return f(*args) + n
        return wrapper
    return decorator

@add_n(5)
def foo():
    return 10

x = foo()
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 15


def test_functools_lru_cache():
    # We need to expose functools to the agent
    agent = Agent()
    agent.module(functools)

    program = """
import functools

# Use a mutable container to verify caching (avoiding 'global')
stats = {"calls": 0}

@functools.lru_cache(maxsize=None)
def fib(n):
    stats["calls"] = stats["calls"] + 1
    if n < 2:
        return n
    return fib(n-1) + fib(n-2)

# First call chain
res1 = fib(5) # 0,1,1,2,3,5

# calls should be 6 (0,1,2,3,4,5 computed once each)
calls_after_first = stats["calls"]

# Second call - should hit cache completely
res2 = fib(5)
calls_after_second = stats["calls"]
"""
    state = eval_and_get_state(program, agent=agent)
    assert state.get("res1") == 5
    assert state.get("res2") == 5
    assert state.get("calls_after_first") == 6
    assert state.get("calls_after_second") == 6

    # Verify cache info can be inspected
    fib_fn = state.get("fib")
    assert hasattr(fib_fn, "cache_info")
    info = fib_fn.cache_info()
    assert info.hits > 0


def test_functools_wraps_metadata():
    agent = Agent()
    agent.module(functools)

    program = """
import functools

def my_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper

@my_decorator
def my_func():
    "Docstring here"
    return 42

name = my_func.__name__
doc = my_func.__doc__
"""
    state = eval_and_get_state(program, agent=agent)
    # Note: UserFunction.name is "my_func"
    # functools.wraps should copy it to the wrapper
    assert state.get("name") == "my_func"
    assert state.get("doc") == "Docstring here"


def test_decorator_ordering():
    # Verify execution order of decorators (bottom-up application)
    program = """
log = []

def dec1(f):
    log.append("dec1")
    return f

def dec2(f):
    log.append("dec2")
    return f

@dec1
@dec2
def foo():
    pass
"""
    state = eval_and_get_state(program)
    # Applied inner-to-outer: dec2 then dec1
    assert state.get("log") == ["dec2", "dec1"]


def test_class_method_decorator():
    # Currently agex supports basic class definitions.
    # We should verify decorators work on methods if methods are supported.
    # Note: visit_FunctionDef handles methods inside classes too.
    program = """
def double_result(f):
    def wrapper(self, x):
        return f(self, x) * 2
    return wrapper

class MyClass:
    @double_result
    def method(self, x):
        return x

obj = MyClass()
res = obj.method(10)
"""
    # This might fail if method handling (self binding) interacts poorly with decorators
    # But let's see.
    state = eval_and_get_state(program)
    assert state.get("res") == 20


def test_unsupported_class_decorator_raises_error():
    """Test that non-@dataclass class decorators raise an error."""
    import pytest

    from agex.eval.error import EvalError

    def my_class_decorator(cls):
        # This would normally modify the class
        return cls

    agent = Agent()
    agent.fn(my_class_decorator)

    program = """
@my_class_decorator
class MyClass:
    pass
"""

    with pytest.raises(EvalError) as e:
        eval_and_get_state(program, agent)
    assert "Class decorators are not supported" in str(e.value)
    assert "Function decorators ARE supported" in str(e.value)


def test_dataclass_decorator_is_allowed():
    """Test that the special @dataclass decorator still works."""
    agent = Agent()

    program = """
@dataclass
class Point:
    x: int
    y: int

p = Point(1, 2)
result = p.x
"""

    state = eval_and_get_state(program, agent)
    assert state.get("result") == 1


def test_multiple_class_decorators_raises_error():
    """Test that multiple decorators on a class raise an error."""
    import pytest

    from agex.eval.error import EvalError

    agent = Agent()

    program = """
@dataclass
@dataclass
class MyClass:
    x: int
"""

    with pytest.raises(EvalError) as e:
        eval_and_get_state(program, agent)
    assert "single @dataclass decorator" in str(e.value)
