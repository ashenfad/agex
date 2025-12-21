import functools

from agex.agent import Agent
from agex.render.view import view


def test_view_robustness_non_callable_registered_as_fn():
    """
    Regression test: ensures that if a non-callable object (like a cached_property,
    or just a plain int) is incorrectly registered as a function (or found as one
    by policy), the view generation does not crash with TypeError from inspect.signature.
    """
    agent = Agent()

    # Manual injection to simulate the problematic state.
    # We want to force 'not_a_function' to be treated as a visible function in __main__
    # so that _render_function is called on it.

    # 1. Create a dummy object that might trick some checks or just use an int
    # functools.cached_property is the specific culprit, so let's try to mimic that
    # behavior if possible, or just use an int which definitely raises TypeError on signature()

    class BrokenDescriptor:
        def __init__(self):
            self.val = 42

    # Create the virtual namespace entry manually to bypass registration checks
    # that might prevent this specific misuse if we used correct API.
    # However, we can use agent._policy.register_fn with a fake callable that is actually not?
    # No, register_fn expects a callable.

    # Let's directly manipulate the policy internals to reproduce the exact state
    # where a name in `main_ns.fns` points to a non-callable object.

    agent._policy.register_fn(func=lambda: None, name="broken_fn")
    # Now swap the object in the registry with something non-callable
    main_ns = agent._policy.namespaces["__main__"]
    main_ns.fn_objects["broken_fn"] = 12345  # an int is not callable

    # Also try directly with a cached_property if possible, though that's usually on a class.
    # The CI failure happened because numpy exposed a cached_property as a module member somehow,
    # or the policy logic found it and thought it was a routine.

    # 2. Render view
    view_output = view(agent)

    # 3. Assertions
    # It should not have crashed.
    # It should contain the fallback signature for the broken item.
    assert "def broken_fn(...):" in view_output
    # And since we didn't give it a docstring and default vis is high, it might try to render docstring
    # Ints don't have __doc__, so it should be fine.


def test_view_robustness_cached_property_on_class():
    """
    Test specifically for cached_property behavior on a class, which was the likely
    source of the issue if it was a class member.
    """

    class HasCachedProp:
        @functools.cached_property
        def prop(self):
            return 42

    agent = Agent()
    # Register the class
    agent.cls(HasCachedProp, name="HasCachedProp")

    # Generate view
    view_output = view(agent)

    # Inspecting a cached_property on a class usually sees it as a descriptor (object),
    # not a function. But let's verify it doesn't crash.
    assert "class HasCachedProp:" in view_output
    # Depending on how it's resolved, it might be shown or not.
    # If it's shown, it shouldn't crash.
