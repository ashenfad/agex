from .helpers import eval_and_get_state


def test_if_elif_else_statement():
    program = """
a = 5
x = 0
if a > 10:
    x = 1
elif a > 7:
    x = 2
elif a > 4:
    x = 3
else:
    x = 4
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 3


def test_for_loop():
    program = """
total = 0
for i in [1, 2, 3, 4]:
    total = total + i
"""
    state = eval_and_get_state(program)
    assert state.get("total") == 10


def test_for_loop_with_destructuring():
    program = """
total = 0
for k, v in [("a", 1), ("b", 2)]:
    total = total + v
"""
    state = eval_and_get_state(program)
    assert state.get("total") == 3


def test_while_loop():
    program = """
i = 0
total = 0
while i < 5:
    i = i + 1
    total = total + i
"""
    state = eval_and_get_state(program)
    assert state.get("total") == 15


def test_loop_break():
    program = """
total = 0
for i in range(100):
    total = total + i
    if i == 4:
        break
"""
    state = eval_and_get_state(program)
    assert state.get("total") == 10  # 0+1+2+3+4


def test_loop_continue():
    program = """
total = 0
for i in range(5): # 0,1,2,3,4
    if i % 2 == 0:
        continue
    total = total + i
"""
    state = eval_and_get_state(program)
    assert state.get("total") == 4  # 1+3


def test_for_loop_else():
    program = """
x = 0
for i in [1, 2, 3]:
    pass
else:
    x = 1

y = 0
for i in [1, 2, 3]:
    if i == 2:
        break
else:
    y = 1
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 1
    assert state.get("y") == 0


def test_while_loop_else():
    program = """
i = 0
x = 0
while i < 3:
    i = i + 1
else:
    x = 1

j = 0
y = 0
while j < 3:
    j = j + 1
    if j == 2:
        break
else:
    y = 1
"""
    state = eval_and_get_state(program)
    assert state.get("x") == 1
    assert state.get("y") == 0
