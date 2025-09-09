from agex import Agent
from agex.eval.core import evaluate_program
from agex.state import Live, Versioned


def test_tuple_to_dict_keys_live_state():
    agent = Agent()
    s = Live()
    # Prepare dictionaries in state
    evaluate_program("state = {}; obj = {'x': 0, 'y': 0}", agent, s)
    # Assign to dict keys via destructuring
    code = "state['a'], state['b'] = 1, 2\nobj['x'], obj['y'] = 3, 4"
    evaluate_program(code, agent, s)
    assert s.get("state") == {"a": 1, "b": 2}
    assert s.get("obj") == {"x": 3, "y": 4}


def test_tuple_to_attributes_live_state():
    agent = Agent()
    s = Live()
    # Create a simple object with attributes by using a class-like dict wrapper
    evaluate_program(
        """
class Box:
    pass
box = Box()
box.a = 0
box.b = 0
""",
        agent,
        s,
    )
    evaluate_program("box.a, box.b = 5, 6\nra=box.a\nrb=box.b", agent, s)
    assert s.get("ra") == 5
    assert s.get("rb") == 6


def test_nested_destructuring_mixed_targets_live_state():
    agent = Agent()
    s = Live()
    evaluate_program("state = {'b': 0}\nclass C: pass\nobj = C()\nobj.x = 0", agent, s)
    code = "(state['a'], (obj.x, state['b'])) = (1, (2, 3))"
    evaluate_program(code + "\nrx=obj.x", agent, s)
    assert s.get("state") == {"a": 1, "b": 3}
    assert s.get("rx") == 2


def test_tuple_to_dict_keys_versioned_state_pickle_safety():
    agent = Agent()
    s = Versioned()
    # Non-picklable placeholder (open file handle) used transiently
    evaluate_program("state = {}", agent, s)
    # This should raise due to Versioned pickle enforcement when assigning file handle
    try:
        evaluate_program(
            "import io\nstate['fh'], state['b'] = io.StringIO(), 2", agent, s
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            "Expected an exception for unpicklable assignment in Versioned state"
        )
