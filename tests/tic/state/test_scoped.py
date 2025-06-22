from tic.state import Ephemeral, Scoped


def test_scoped_state_read_through():
    parent = Ephemeral()
    parent.set("x", 10)

    scoped = Scoped(parent)
    assert scoped.get("x") == 10


def test_scoped_state_write_is_local():
    parent = Ephemeral()
    parent.set("x", 10)

    scoped = Scoped(parent)
    scoped.set("y", 20)
    scoped.set("x", 99)  # Shadow the parent variable

    assert scoped.get("y") == 20
    assert scoped.get("x") == 99  # Should get the local value
    assert parent.get("x") == 10  # Parent should be unchanged
    assert "y" not in parent


def test_scoped_state_simulates_late_binding():
    """
    This test proves that the Scoped state correctly simulates Python's
    late-binding closures by holding a reference to the parent scope, not a copy.
    """
    parent = Ephemeral()
    parent.set("x", 10)

    # The 'closure' is created, capturing a reference to the parent state
    closure_scope = Scoped(parent)

    # At this point, something happens in the parent scope
    parent.set("x", 99)

    # When the 'closure' is finally used, it should see the new value
    assert closure_scope.get("x") == 99
