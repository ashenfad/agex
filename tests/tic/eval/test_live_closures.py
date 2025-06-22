from tests.tic.eval.helpers import eval_and_get_state
from tic.state.ephemeral import Ephemeral
from tic.state.kv import Memory
from tic.state.versioned import Versioned


def test_basic_closure_and_late_binding():
    """
    Tests that a closure can access parent variables and that it sees
    updates made after its definition (late-binding).
    """
    program = """
x = 10
def get_x_factory():
    def inner_get_x():
        return x
    return inner_get_x

get_x = get_x_factory()
result1 = get_x() # Should see the initial value of x

x = 20 # Update x
result2 = get_x() # Should see the new value of x
"""
    state = eval_and_get_state(program)
    assert state.get("result1") == 10
    assert state.get("result2") == 20


def test_closure_freezing_on_snapshot():
    """
    Tests that a closure's late-binding is 'frozen' at the moment
    a snapshot is taken.
    """
    # 1. Setup initial state and the function with a closure
    program1 = """
x = 100
def get_x():
    return x
"""
    kv_store = Memory(as_bytes=False)
    state = Versioned(kv_store)
    eval_and_get_state(program1, state=state)

    # 2. Verify initial behavior
    result1 = eval_and_get_state("res = get_x()", state=state).get("res")
    assert result1 == 100

    # 3. Update the closed-over variable and take a snapshot
    eval_and_get_state("x = 200", state=state)
    commit_hash = state.snapshot()

    # 4. Verify that the live function sees the update
    result2 = eval_and_get_state("res = get_x()", state=state).get("res")
    assert result2 == 200

    # 5. Create a NEW versioned state from the snapshot
    restored_state = Versioned(kv_store, commit_hash=commit_hash)

    # 6. Verify the restored function has the frozen value (200)
    result3 = eval_and_get_state("res = get_x()", state=restored_state).get("res")
    assert result3 == 200

    # 7. Update x in the new state and confirm the frozen function does NOT see it
    eval_and_get_state("x = 300", state=restored_state)
    result4 = eval_and_get_state("res = get_x()", state=restored_state).get("res")
    assert result4 == 200  # Still 200, because the closure is frozen


def test_closure_on_non_versioned_state():
    """
    Make sure closures still work fine on a simple Ephemeral state
    that doesn't get versioned.
    """
    program = """
x = 1
def f():
    def g():
        return x
    return g

fn = f()
res1 = fn()

x = 5
res2 = fn()
"""
    state = Ephemeral()
    eval_and_get_state(program, state=state)
    assert state.get("res1") == 1
    assert state.get("res2") == 5


def test_nested_closures():
    """Tests that closures work correctly through multiple levels of nesting."""
    program = """
x = 10
def f1():
    y = 20
    def f2():
        z = 30
        def f3():
            return x + y + z
        return f3
    return f2

get_sum = f1()()
result = get_sum()
"""
    state = eval_and_get_state(program)
    assert state.get("result") == 60


def test_closure_with_no_free_variables():
    """Tests that a function with no free variables is handled correctly."""
    program = """
def simple():
    a = 1
    b = 2
    return a + b

result = simple()
"""
    state = eval_and_get_state(program)
    assert state.get("result") == 3


def test_multiple_independent_closures():
    """
    Tests that two functions closing over the same scope both see updates.
    """
    program = """
x = 100
def factory():
    def get_x():
        return x
    def get_x_plus_one():
        return x + 1
    return get_x, get_x_plus_one

f1, f2 = factory()

res1 = f1()
res2 = f2()

x = 200 # update the free variable

res3 = f1()
res4 = f2()
"""
    state = eval_and_get_state(program)
    assert state.get("res1") == 100
    assert state.get("res2") == 101
    assert state.get("res3") == 200
    assert state.get("res4") == 201


def test_closure_with_shadowing():
    """
    Tests that a local variable correctly shadows a free variable.
    The analyzer should not identify the shadowed variable as 'free'.
    """
    program = """
x = 10 # This should be shadowed
def f():
    x = 20 # This is the 'x' that should be seen
    def g():
        return x
    return g

fn = f()
result = fn()
"""
    state = eval_and_get_state(program)
    assert state.get("result") == 20
