import collections
import math
import pickle
import sqlite3
from dataclasses import dataclass
from types import ModuleType

import pytest

from agex import events
from agex.agent import Agent, MemberSpec
from agex.agent.base import clear_agent_registry
from agex.agent.datatypes import UnpicklableMarker
from agex.agent.events import ActionEvent, OutputEvent, SuccessEvent, TaskStartEvent
from agex.agent.policy.describe import (
    describe_class,
    describe_member,
    describe_namespace,
)
from agex.llm import Dummy
from agex.llm.core import LLMResponse
from agex.state import Namespaced, Versioned, connect_state
from tests.agex import test_module


def test_view_image_primer_text_is_always_visible():
    """
    Tests that the core primer text is always included.
    """
    # Agent with NO image-related modules
    agent = Agent()
    system_message = agent._build_system_message()
    # Check for core philosophy items
    assert "Code is Action" in system_message
    assert "Persistent State" in system_message
    assert "Task Control Functions" in system_message
    assert "task_continue" in system_message


def test_agent_fn_registration_decorator():
    agent = Agent()

    @agent.fn()
    def my_func():
        """My docstring"""
        return 1

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "my_func" in main.fns
    ms = main.fns["my_func"]
    assert ms.visibility == "high"
    assert ms.docstring == "My docstring"


def test_agent_fn_registration_decorator_with_args():
    agent = Agent()

    @agent.fn(visibility="low", docstring="New doc")
    def my_func():
        """Original doc"""
        return 1

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "my_func" in main.fns
    ms = main.fns["my_func"]
    assert ms.visibility == "low"
    assert ms.docstring == "New doc"


def test_agent_fn_registration_functional():
    agent = Agent()

    def my_func():
        return 1

    agent.fn(visibility="medium")(my_func)

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "my_func" in main.fns
    ms = main.fns["my_func"]
    assert ms.visibility == "medium"
    assert (ms.docstring or None) is None


def test_agent_fn_registration_functional_builtin():
    agent = Agent()
    agent.fn()(math.sqrt)  # Test the decorator factory style
    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "sqrt" in main.fns


def test_agent_fn_registration_direct_call():
    agent = Agent()
    agent.fn(math.sqrt)  # Test the direct call style
    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "sqrt" in main.fns


def test_context_manager_bound_variable_is_cleaned_up():
    """Context manager bindings (e.g., 'as conn') should not persist after the block."""
    clear_agent_registry()

    llm = Dummy(
        [
            LLMResponse(
                thinking="Create a table via sqlite3 context manager.",
                code="""with db as conn:
    conn.execute("CREATE TABLE IF NOT EXISTS users(id INTEGER)")
task_success("done")""",
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="db_agent", llm=llm, state=config)
    connection = sqlite3.connect(":memory:")
    agent.module(connection, name="db", include=["execute", "commit"])

    @agent.task
    def create_table() -> str:  # type: ignore[return-value]
        """Create a table using sqlite3 context manager."""
        ...

    result = create_table(session="test_session")
    assert result == "done"
    state = agent._host.resolve_state(config, "test_session")
    assert "db_agent/conn" not in state.keys()
    connection.close()


def test_helper_recap_skips_unpicklable_markers():
    """UserFunction recap should skip state entries that raise UnpicklableVariableError."""
    clear_agent_registry()

    llm = Dummy([LLMResponse(thinking="Simple completion.", code='task_success("ok")')])
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="marker_agent", llm=llm, state=config)

    @agent.task
    def marker_task() -> str:  # type: ignore[return-value]
        """Return without defining helpers."""
        ...

    state = agent._host.resolve_state(config, "test_session")
    marker = UnpicklableMarker(
        variable_name="marker_agent/bad",
        type_name="BadObject",
        original_exception="cannot pickle",
    )
    state.set("marker_agent/bad", marker)
    state.snapshot()

    assert marker_task(session="test_session") == "ok"


def test_agent_fn_registration_with_name_alias():
    agent = Agent()

    def original_function_name():
        return "aliased"

    agent.fn(original_function_name, name="alias")

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "alias" in main.fns
    assert "original_function_name" not in main.fns


def test_registering_reserved_name_fails():
    """Tests that registering a reserved name raises a ValueError."""
    agent = Agent()

    def my_fn():
        pass

    class MyClass:
        pass

    dummy_module = ModuleType("dummy")

    with pytest.raises(ValueError, match="is reserved"):
        agent.fn(my_fn, name="dataclass")

    with pytest.raises(ValueError, match="is reserved"):
        agent.cls(MyClass, name="dataclass")

    with pytest.raises(ValueError, match="is reserved"):
        agent.module(dummy_module, name="dataclasses")


def test_agent_cls_registration_defaults():
    agent = Agent()

    @agent.cls
    @dataclass
    class MyData:
        x: int
        y: str
        _z: float

        def do_stuff(self):
            pass

        def _do_private_stuff(self):
            pass

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "MyData" in main.classes
    rc = main.classes["MyData"]
    ns = agent._policy._class_namespaces[rc.cls]
    desc = describe_class(rc.cls, ns, include_low=True)
    assert desc.constructable is True
    assert "do_stuff" in (desc.members or {})
    assert "__init__" in (desc.members or {})
    assert "_do_private_stuff" not in (desc.members or {})
    # attributes
    members = desc.members or {}
    assert "x" in members and "y" in members and "_z" not in members


def test_agent_cls_registration_selectors():
    agent = Agent()

    class MyClass:
        x: int
        _y: str

        def do_stuff(self):
            pass

        def _do_private_stuff(self):
            pass

    # Use as a decorator factory
    agent.cls(
        MyClass,
        visibility="medium",
        constructable=False,
        include=["x", "do_stuff", "_y"],
        exclude=None,  # Include everything from the list
    )

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "MyClass" in main.classes
    rc = main.classes["MyClass"]
    ns = agent._policy._class_namespaces[rc.cls]
    desc = describe_class(rc.cls, ns, include_low=True)
    assert desc.constructable is False
    members = desc.members or {}
    assert set(k for k, v in members.items() if v.kind == "obj") == {"x", "_y"}
    assert "do_stuff" in members and "__init__" not in members


def test_agent_cls_registration_configure_and_exclude():
    agent = Agent()

    class MyService:
        config_path = "/etc/service.conf"
        name: str = "default_name"
        _internal_id = "xyz-123"

        def __init__(self):
            pass

        def critical_op(self):
            pass

        def regular_op(self):
            pass

        def _private_op(self):
            pass

    agent.cls(
        MyService,
        visibility="medium",  # Default for selected
        include="*",  # Explicitly include everything to start
        exclude=["regular_op", "_*"],  # Exclude one public and all private
        configure={
            "critical_op": MemberSpec(visibility="high"),
            "config_path": MemberSpec(visibility="low"),
        },
    )

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    rc = main.classes["MyService"]
    ns = agent._policy._class_namespaces[rc.cls]
    desc = describe_class(rc.cls, ns, include_low=True)
    members = desc.members or {}
    # methods
    assert "critical_op" in members and members["critical_op"].visibility == "high"
    assert "__init__" in members
    assert "_private_op" not in members
    # attrs
    assert "config_path" in members and members["config_path"].visibility == "low"
    assert "name" in members


def test_agent_module_registration():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",
        include=["public_fn", "PI", "PublicClass", "PublicClass.*"],
        exclude=["*.secret_*", "*._*"],
        configure={
            "PI": MemberSpec(visibility="high"),
            "PublicClass": MemberSpec(constructable=True),  # Ensure it's constructable
            "PublicClass.public_method": MemberSpec(visibility="high"),
        },
    )

    ns = agent._policy.namespaces.get("sample")
    assert ns is not None and ns.kind == "module"
    desc = describe_namespace(ns, include_low=True)
    # Top-level
    assert desc["PI"].visibility == "high"
    assert desc["public_fn"].visibility == "low"
    # Class
    cdesc = describe_class(
        getattr(ns._ensure_module_loaded(), "PublicClass"), ns, include_low=True
    )
    assert cdesc.constructable is True
    m = describe_member(ns, "PublicClass.public_method")
    assert m is not None and m.visibility == "high"


def test_agent_module_registration_defaults():
    agent = Agent()
    agent.module(test_module, name="test_module")

    ns = agent._policy.namespaces.get("test_module")
    assert ns is not None and ns.kind == "module"
    desc = describe_namespace(ns, include_low=True)
    assert "public_fn" in desc and desc["public_fn"].kind == "fn"
    assert "PI" in desc and desc["PI"].kind == "obj"
    assert "PublicClass" in desc and desc["PublicClass"].kind == "class"
    assert "_PrivateClass" not in desc
    # Validate nested member via describe_member
    m = describe_member(ns, "PublicClass.public_method")
    assert m is not None and m.kind == "fn"


def test_agent_module_with_configure():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",  # default for selected items
        include=["*"],
        exclude=["_*", "*._*"],  # Exclude both top-level and class privates
        configure={
            "PI": MemberSpec(visibility="high"),
            "PublicClass": MemberSpec(constructable=False),
            "PublicClass.public_method": MemberSpec(visibility="high"),
            "public_fn": MemberSpec(visibility="medium"),
        },
    )

    ns = agent._policy.namespaces.get("sample")
    assert ns is not None and ns.kind == "module"
    desc = describe_namespace(ns, include_low=True)
    assert desc["public_fn"].visibility == "medium"
    assert desc["PI"].visibility == "high"
    # class constructable and nested method vis
    cdesc = describe_class(
        getattr(ns._ensure_module_loaded(), "PublicClass"), ns, include_low=True
    )
    assert cdesc.constructable is False
    m = describe_member(ns, "PublicClass.public_method")
    assert m is not None and m.visibility == "high"


def test_agent_cls_no_parens():
    """Tests that the @agent.cls decorator works without parentheses."""
    agent = Agent()

    @agent.cls
    @dataclass
    class SimpleData:
        value: str

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "SimpleData" in main.classes


def test_agent_cls_direct_call():
    agent = Agent()

    class MyClass:
        pass

    agent.cls(MyClass, visibility="low")

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "MyClass" in main.classes


def test_agent_cls_with_name_alias():
    agent = Agent()

    class OriginalClassName:
        pass

    agent.cls(OriginalClassName, name="AliasClass")  # type: ignore

    main = agent._policy.namespaces.get("__main__")
    assert main is not None
    assert "AliasClass" in main.classes
    assert "OriginalClassName" not in main.classes


# =============================================================================
# Decorator Validation Tests
# =============================================================================


def test_task_decorator_single():
    """Test that a single task decorator works correctly."""
    agent = Agent()

    @agent.task("Implement a simple function")
    def simple_task():
        """A simple task function."""
        pass

    # Check that task metadata is set correctly
    assert hasattr(simple_task, "__agex_task_namespace__")
    assert simple_task.__agex_task_namespace__ == agent.name


def test_fn_decorator_multiple():
    """Test that multiple fn decorators on the same function are allowed."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.fn(docstring="First agent function")
    @agent2.fn(docstring="Second agent function")
    def shared_function():
        """A function shared across agents."""
        return "shared"

    # Check that fn metadata is set correctly
    assert hasattr(shared_function, "__is_agent_fn__")
    assert shared_function.__is_agent_fn__ is True
    assert hasattr(shared_function, "__agent_fn_owners__")
    assert len(shared_function.__agent_fn_owners__) == 2
    assert agent1 in shared_function.__agent_fn_owners__
    assert agent2 in shared_function.__agent_fn_owners__


def test_task_decorator_multiple_not_allowed():
    """Test that multiple task decorators on the same function are not allowed."""
    agent1 = Agent()
    agent2 = Agent()

    # First task decorator should work
    @agent1.task("First task implementation")
    def multi_task_attempt():
        pass

    # Second task decorator should fail
    with pytest.raises(ValueError, match="already has a task decorator"):
        agent2.task("Second task implementation")(multi_task_attempt)


def test_decorator_order_wrong_not_allowed():
    """Test that wrong decorator order (task before fn) is not allowed."""
    agent1 = Agent()
    agent2 = Agent()

    with pytest.raises(ValueError, match="Invalid decorator order"):

        @agent1.task("This should fail")  # Task applied first (inner)
        @agent2.fn(
            docstring="This comes after"
        )  # Fn applied second (outer) - WRONG ORDER
        def wrong_order_example():
            pass


def test_decorator_order_correct_allowed():
    """Test that correct decorator order (fn before task) is allowed."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.fn(docstring="Outer fn decorator")  # Fn applied first (outer) - CORRECT
    @agent2.task("Inner task decorator")  # Task applied second (inner) - CORRECT
    def correct_order_example():
        """Function with correct dual decorator order."""
        pass

    # Check that both decorators were applied
    assert hasattr(correct_order_example, "__agex_task_namespace__")
    assert (
        correct_order_example.__agex_task_namespace__ == agent2.name
    )  # agent2's class name


def test_fn_decorator_builtin_functions():
    """Test that fn decorator works with built-in functions without errors."""
    agent = Agent()

    # This should not raise any AttributeError about setting __agent_fn_owners__
    registered_sqrt = agent.fn(docstring="Built-in square root")(math.sqrt)

    # Should be the same function object
    assert registered_sqrt is math.sqrt

    # Should be registered in the agent
    main = agent._policy.namespaces.get("__main__")
    assert main is not None and "sqrt" in main.fns


def test_task_decorator_validation_error_messages():
    """Test that validation error messages are clear and helpful."""
    agent1 = Agent()
    agent2 = Agent()

    @agent1.task("First task")
    def test_function():
        pass

    # Test multiple task decorator error message
    with pytest.raises(ValueError) as exc_info:
        agent2.task("Second task")(test_function)

    error_msg = str(exc_info.value)
    assert "already has a task decorator" in error_msg
    assert "Multi-agent tasks are not supported" in error_msg

    # Test wrong order error message
    with pytest.raises(ValueError) as exc_info:

        @agent1.task("Should fail")
        @agent2.fn(docstring="Wrong order")
        def wrong_order():
            pass

    error_msg = str(exc_info.value)
    assert "Invalid decorator order" in error_msg
    assert "@agent.fn() must be applied AFTER @agent.task()" in error_msg
    assert "Correct order:" in error_msg


def test_fn_registration_of_own_task_not_allowed():
    """Test that registering a task as a capability on the same agent raises an error."""
    agent = Agent()

    @agent.task("Do something")
    def my_task():
        """A task function."""
        pass

    # Attempt to register the same task as a capability on the same agent
    with pytest.raises(ValueError) as exc_info:
        agent.fn(my_task)

    error_msg = str(exc_info.value)
    assert "Cannot register" in error_msg
    assert "same agent" in error_msg
    assert "Task functions are automatically available" in error_msg


def test_agent_names_and_uniqueness():
    """Test agent name assignment and uniqueness enforcement."""
    # Clear registry for clean test
    clear_agent_registry()

    # Test agent creation with names
    agent1 = Agent(name="test_agent")
    agent2 = Agent(name="other_agent")

    assert agent1.name == "test_agent"
    assert agent2.name == "other_agent"

    # Test duplicate name prevention

    # 1. Identical agent (same fingerprint) should NOT raise (simulates deserialization)
    Agent(name="test_agent")

    # 2. Different agent (different fingerprint) SHOULD raise
    with pytest.raises(ValueError, match="Agent name 'test_agent' already exists"):
        Agent(name="test_agent", primer="Different instructions")


def test_dual_decorator_namespace_setting():
    """Test that dual-decorated functions get proper namespace metadata."""
    clear_agent_registry()

    # Create agents with names
    orchestrator = Agent(name="orchestrator")
    specialist = Agent(name="specialist")

    # Create dual-decorated function
    @orchestrator.fn(docstring="Specialist utility")
    @specialist.task("Perform specialized task")
    def dual_function():
        """A dual-decorated function."""
        pass

    # Verify namespace is set correctly
    assert hasattr(dual_function, "__agex_task_namespace__")
    assert dual_function.__agex_task_namespace__ == "specialist"

    # Verify it's registered in the fn decorator's agent via policy
    main = orchestrator._policy.namespaces.get("__main__")
    assert main is not None and "dual_function" in main.fns

    # Verify dual-decorator metadata (namespace is sufficient)
    # The __agex_task_namespace__ attribute serves as both the task marker and namespace


def test_namespaced_state_isolation():
    """Test that Namespaced state provides proper isolation."""
    # Create shared state
    main_state = Versioned()

    # Create namespaced views
    namespace_a = Namespaced(main_state, "agent_a")
    namespace_b = Namespaced(main_state, "agent_b")

    # Set namespace-specific data
    namespace_a.set("local_data", "value from A")
    namespace_b.set("local_data", "value from B")

    # Verify isolation - each namespace only sees its own data
    assert namespace_a.get("local_data") == "value from A"
    assert namespace_b.get("local_data") == "value from B"
    assert namespace_a.get("local_data") != namespace_b.get("local_data")

    # Verify namespaces don't see each other's data
    assert namespace_a.get("nonexistent") is None
    assert namespace_b.get("nonexistent") is None

    # Verify the underlying state has the namespaced keys
    assert "agent_a/local_data" in main_state
    assert "agent_b/local_data" in main_state
    assert main_state.get("agent_a/local_data") == "value from A"
    assert main_state.get("agent_b/local_data") == "value from B"


def test_task_input_dataclass_pickling():
    """Test that task input dataclasses can be pickled and snapshotted."""
    clear_agent_registry()

    # Create agent with dummy LLM client to avoid real API calls
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="I will return the expected result",
                code="task_success('test result')",
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="test_agent", llm=llm, state=config)

    @agent.task("Test task with inputs")
    def test_task(message: str, value: int) -> str:  # type: ignore
        """A test task with parameters."""
        pass

    # This will trigger creation and storage of the input dataclass
    result = test_task(message="hello", value=42, session="test_session")
    assert result == "test result"  # Verify the dummy LLM response was used

    # Verify inputs were stored and are pickleable
    state = agent._host.resolve_state(config, "test_session")
    inputs = state.get("inputs")  # No longer namespaced
    assert inputs is not None
    assert inputs.message == "hello"
    assert inputs.value == 42

    # Test direct pickling of the inputs instance
    pickled_inputs = pickle.dumps(inputs)
    unpickled_inputs = pickle.loads(pickled_inputs)
    assert unpickled_inputs.message == "hello"
    assert unpickled_inputs.value == 42

    # Test state snapshotting (which internally pickles all state data)
    snapshot_hash = state.snapshot().commit_hash
    assert snapshot_hash is not None
    assert len(snapshot_hash) > 0


def test_unserializable_object_in_state_is_handled_gracefully():
    """
    Test that if an unserializable object (like a lambda) is added to state
    via mutation, the snapshot process handles it gracefully by creating a marker.
    The agent won't see any warnings (silent success), but will get a clear error
    if they try to access the variable in a future turn.
    """

    clear_agent_registry()

    # This fn mutates a dictionary to include a real Python lambda,
    # making the dictionary unserializable.
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="Mutating an object to make it unserializable.",
                code="make_object_unserializable(my_object)",
            ),
            LLMResponse(
                thinking="Now I will finish.",
                code="task_success('done')",
            ),
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="test_agent", llm=llm, state=config)

    class Unserializable:
        def __getstate__(self):
            raise pickle.PicklingError("This object cannot be pickled.")

    @agent.fn()
    def make_object_unserializable(obj):
        obj["bad_field"] = Unserializable()

    @agent.task("A task that creates bad state via mutation.")
    def task_with_unserializable_state() -> str:  # type: ignore
        """This task will create unserializable state by mutation."""
        pass

    # Pre-populate the state with a serializable object.
    state = agent._host.resolve_state(config, "test_session")
    state.set("my_object", {"a": 1})  # No longer namespaced
    state.snapshot()

    # Run the task. This will mutate my_object and then try to snapshot.
    # It should NOT raise a PicklingError - a marker is created instead.
    result = task_with_unserializable_state(session="test_session")  # type: ignore
    assert result == "done"

    # With the new marker system, there's NO warning shown (silent success)
    # The marker is successfully saved in place of the unpicklable object

    # Verify that trying to access the variable would now raise an error
    from agex.agent.datatypes import UnpicklableVariableError

    # Directly check that the variable is now a marker
    with pytest.raises(UnpicklableVariableError) as exc_info:
        state.get("my_object")  # No longer namespaced

    # Verify the error message is helpful
    error_msg = str(exc_info.value)
    assert "my_object" in error_msg
    assert "not available" in error_msg
    assert "unpicklable" in error_msg
    assert "Solutions:" in error_msg


def test_shallow_validation_on_large_input_list():
    """
    Tests that the shallow validator catches bad data in a large input list.
    """

    clear_agent_registry()
    # The non-failing path of this test will enter the task loop.
    # We provide a single dummy response for it to consume.
    llm = Dummy(responses=[LLMResponse(thinking="Looks good.", code="task_success(1)")])
    agent = Agent(name="test_agent", llm=llm)

    @agent.task("A task that accepts a large list.")
    def process_large_list(items: list[int]) -> int:  # type: ignore
        pass

    good_list = list(range(2000))
    bad_list = list(range(2000))
    bad_list[-5] = "not a number"  # type: ignore

    # This should pass. The state kwarg is added by the decorator.
    process_large_list(items=good_list)  # type: ignore

    # This should fail validation
    with pytest.raises(ValueError) as exc_info:
        process_large_list(items=bad_list)  # type: ignore

    error_msg = str(exc_info.value)
    assert "Validation failed for argument 'items'" in error_msg
    assert "Input should be a valid integer" in error_msg


def test_shallow_validation_on_agent_output():
    """
    Tests that the agent gets feedback if its output doesn't match the
    return type annotation, especially for large collections.
    """

    clear_agent_registry()

    # Large, valid dictionary
    large_valid_dict = {f"key_{i}": i for i in range(150)}
    # Large, invalid dictionary (error in the tail)
    large_invalid_dict = large_valid_dict.copy()
    large_invalid_dict["key_145"] = "not an int"  # type: ignore

    llm = Dummy()
    # Using side_effect to ensure responses are consumed sequentially, even if the loop retries multiple times
    llm.responses = [
        LLMResponse(
            thinking="I will try to return an invalid dictionary.",
            code="task_success(invalid_dict)",
        ),
        LLMResponse(
            thinking="That failed. I will return a valid dictionary now.",
            code="task_success(valid_dict)",
        ),
    ]
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="test_agent", llm=llm, state=config, max_iterations=10)

    @agent.task("A task that returns a large dictionary.")
    def produce_large_dict() -> dict[str, int]:  # type: ignore
        pass

    # Pre-populate state to avoid parsing large literals in the agent's code
    state = agent._host.resolve_state(config, "test_session")
    state.set("invalid_dict", large_invalid_dict)  # No longer namespaced
    state.set("valid_dict", large_valid_dict)  # No longer namespaced

    result = produce_large_dict(session="test_session")  # type: ignore

    # Check that the final result is the valid one
    assert result == large_valid_dict

    # Check that validation error was shown to agent as output
    # The agent DID see the validation error (as evidenced by the fact that it then provided valid output)
    all_events = [e for e in events(state) if e.full_namespace == "test_agent"]
    output_events = [e for e in all_events if isinstance(e, OutputEvent)]

    # Find output events that contain validation error messages
    validation_outputs = []
    for event in output_events:
        for part in event.parts:
            # Handle both raw PrintAction objects and rendered TextPart objects
            part_text = ""
            if hasattr(part, "text"):
                part_text = str(part.text)
            elif hasattr(part, "__iter__") and len(part) > 0:
                # PrintAction is iterable, get the first argument
                part_text = str(part[0])

            if "Output validation failed" in part_text:
                validation_outputs.append(event)
                break

    assert len(validation_outputs) >= 1

    # Get the error message from the PrintAction or TextPart
    error_part = validation_outputs[0].parts[0]
    if hasattr(error_part, "text"):
        error_message = str(error_part.text)
    else:
        # It's a PrintAction, get the first argument
        error_message = str(error_part[0])

    assert "Output validation failed" in error_message
    assert "dict[str, int]" in error_message
    assert "..." not in error_message


def test_task_setup_functionality():
    """Test that task setup parameter works correctly."""
    clear_agent_registry()

    # Create agent
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="I can see the setup variable and will complete the task",
                code='task_success(f"Setup value: {setup_var}")',
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="setup_test_agent", llm=llm, state=config)

    # Define task with setup
    @agent.task(primer="Test task with setup", setup='setup_var = "Hello from setup!"')
    def test_task() -> str:  # type: ignore[return-value]
        """Test task with setup"""
        ...

    # Execute task
    result = test_task(session="test_session")

    # Verify result includes setup data
    assert "Setup value: Hello from setup!" in result

    # Verify events
    state = agent._host.resolve_state(config, "test_session")
    event_list = events(state)

    # Should have: TaskStart, Setup ActionEvent, Agent ActionEvent, SuccessEvent
    assert len(event_list) == 4

    # Check setup event
    setup_event = event_list[1]
    assert isinstance(setup_event, ActionEvent)
    assert (
        setup_event.thinking
        == "This code was automatically run to provide context for the task."
    )
    assert setup_event.code == 'setup_var = "Hello from setup!"'

    # Check that setup variable is available in state
    setup_value = state.get("setup_var")  # No longer namespaced
    assert setup_value == "Hello from setup!"


def test_task_setup_error_handling():
    """Test that setup errors are handled gracefully."""
    clear_agent_registry()

    # Create agent
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="I see there was an error in setup, but I can still complete the task",
                code='task_success("completed despite setup error")',
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="setup_error_agent", llm=llm, state=config)

    # Define task with setup that will error
    @agent.task(
        primer="Test task with setup error",
        setup="invalid_variable = undefined_function()",
    )
    def test_task() -> str:  # type: ignore[return-value]
        """Test task with setup that errors"""
        ...

    # Execute task
    result = test_task(session="test_session")

    # Task should still complete
    assert "completed despite setup error" in result

    # Verify events include error
    state = agent._host.resolve_state(config, "test_session")
    event_list = events(state)

    # Should have: TaskStart, Setup ActionEvent, Agent ActionEvent, SuccessEvent
    # (Setup errors might not create separate ErrorEvents)
    assert len(event_list) == 4

    # Check that there's setup and agent action events
    action_events = [e for e in event_list if isinstance(e, ActionEvent)]
    assert len(action_events) == 2  # Setup action + agent action

    # First ActionEvent should be the setup
    setup_event = action_events[0]
    assert (
        setup_event.thinking
        == "This code was automatically run to provide context for the task."
    )
    assert setup_event.code == "invalid_variable = undefined_function()"


def test_task_without_setup():
    """Test that tasks without setup still work normally."""
    clear_agent_registry()

    # Create agent
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="Simple task completion",
                code='task_success("completed without setup")',
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="no_setup_agent", llm=llm, state=config)

    # Define task without setup
    @agent.task(primer="Test task without setup")
    def test_task() -> str:  # type: ignore[return-value]
        """Test task without setup"""
        ...

    # Execute task
    result = test_task(session="test_session")

    # Verify result
    assert "completed without setup" in result

    # Verify events - should NOT have setup event
    state = agent._host.resolve_state(config, "test_session")
    event_list = events(state)

    # Should have: TaskStart, Agent ActionEvent, SuccessEvent
    assert len(event_list) == 3

    # No setup ActionEvent should be present
    action_events = [e for e in event_list if isinstance(e, ActionEvent)]
    assert len(action_events) == 1  # Only the agent's own action
    assert action_events[0].thinking == "Simple task completion"


def test_setup_events_tagged_with_source():
    """Test that setup events are tagged with source='setup'."""
    clear_agent_registry()

    # Create agent that uses task_continue in setup
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="I can see the setup output and complete",
                code='task_success("done")',
            )
        ]
    )
    config = connect_state(type="versioned", storage="memory")
    agent = Agent(name="setup_source_agent", llm=llm, state=config)

    # Define task with setup that creates output
    @agent.task(
        primer="Test source tagging",
        setup='task_continue("Setup context loaded", {"data": [1, 2, 3]})',
    )
    def test_task() -> str:  # type: ignore[return-value]
        """Test task with setup that produces output"""
        ...

    # Execute task
    result = test_task(session="test_session")
    assert result == "done"

    # Verify event source tagging
    state = agent._host.resolve_state(config, "test_session")
    event_list = events(state)

    # Find setup ActionEvent
    setup_action = [
        e
        for e in event_list
        if isinstance(e, ActionEvent) and "automatically run" in e.thinking
    ]
    assert len(setup_action) == 1
    assert setup_action[0].source == "setup"

    # Find setup OutputEvents (from task_continue)
    setup_outputs = [
        e for e in event_list if isinstance(e, OutputEvent) and e.source == "setup"
    ]
    assert len(setup_outputs) >= 1  # At least one from task_continue

    # Find main execution ActionEvent
    main_actions = [
        e
        for e in event_list
        if isinstance(e, ActionEvent) and "automatically run" not in e.thinking
    ]
    assert len(main_actions) == 1
    assert main_actions[0].source == "main"  # Main events have default source

    # Verify TaskStartEvent and SuccessEvent have default source
    task_start = [e for e in event_list if isinstance(e, TaskStartEvent)]
    assert len(task_start) == 1
    assert task_start[0].source == "main"

    success = [e for e in event_list if isinstance(e, SuccessEvent)]
    assert len(success) == 1
    assert success[0].source == "main"


def test_setup_code_event_sequence():
    """Test that setup code produces the expected event sequence."""
    clear_agent_registry()

    # Setup code that produces output to test event ordering
    SETUP_CODE = """
print("Setup is running")
setup_var = "Hello from setup!"
print(f"Setup complete: {setup_var}")
"""

    # Create agent with setup code
    config = connect_state(type="versioned", storage="memory")
    llm = Dummy(
        responses=[
            LLMResponse(
                thinking="I will complete immediately", code='task_success("done")'
            )
        ]
    )
    agent = Agent(name="test_agent", llm=llm, state=config)

    @agent.task(primer="Test task", setup=SETUP_CODE)
    def test_task(prompt: str) -> str:  # type: ignore[return-value]
        """Test task with setup code"""
        ...

    result = test_task("test", session="test_session")
    state = agent._host.resolve_state(config, "test_session")
    event_list = events(state)

    # Verify result
    assert result == "done"

    # Verify expected event sequence
    expected_sequence = [
        TaskStartEvent,  # Task starts
        ActionEvent,  # Setup action
        OutputEvent,  # Setup output 1: "Setup is running"
        OutputEvent,  # Setup output 2: "Setup complete: Hello from setup!"
        ActionEvent,  # Agent action
        SuccessEvent,  # Task completion
    ]

    assert len(event_list) == len(
        expected_sequence
    ), f"Expected {len(expected_sequence)} events, got {len(event_list)}"

    for i, (event, expected_type) in enumerate(zip(event_list, expected_sequence)):
        assert isinstance(
            event, expected_type
        ), f"Event {i} should be {expected_type.__name__}, got {type(event).__name__}"

    # Verify setup ActionEvent is immediately followed by its OutputEvents
    setup_action = event_list[1]
    assert isinstance(setup_action, ActionEvent)
    assert (
        setup_action.thinking
        == "This code was automatically run to provide context for the task."
    )

    # Next events should be OutputEvents from setup
    setup_output_1 = event_list[2]
    setup_output_2 = event_list[3]
    assert isinstance(setup_output_1, OutputEvent)
    assert isinstance(setup_output_2, OutputEvent)

    # Verify output content
    output_1_text = str(setup_output_1.parts[0])
    output_2_text = str(setup_output_2.parts[0])
    assert "Setup is running" in output_1_text
    assert "Setup complete: Hello from setup!" in output_2_text

    # Verify the last ActionEvent is the agent's actual response
    agent_action = event_list[4]
    assert isinstance(agent_action, ActionEvent)
    assert agent_action.thinking == "I will complete immediately"
    assert agent_action.code == 'task_success("done")'


def test_agent_module_registration_with_instance_methods_and_name_override():
    agent = Agent()
    agent.module(
        test_module,
        name="sample",
        visibility="low",
        include=["public_fn", "PI", "PublicClass", "PublicClass.*"],
        exclude=["*.secret_*", "*._*"],
        configure={
            "PI": MemberSpec(visibility="high"),
            "PublicClass": MemberSpec(constructable=True),  # Ensure it's constructable
            "PublicClass.public_method": MemberSpec(visibility="high"),
        },
    )

    ns = agent._policy.namespaces.get("sample")
    assert ns is not None and ns.kind == "module"
    desc = describe_namespace(ns, include_low=True)
    # Top-level
    assert desc["PI"].visibility == "high"
    assert desc["public_fn"].visibility == "low"
    # Class
    cdesc = describe_class(
        getattr(ns._ensure_module_loaded(), "PublicClass"), ns, include_low=True
    )
    assert cdesc.constructable is True
    m = describe_member(ns, "PublicClass.public_method")
    assert m is not None and m.visibility == "high"


def test_recursive_module_registration_allows_submodule_imports():
    """
    Tests that a module registered with recursive=True allows 'from ... import ...'
    for its submodules.
    """
    agent = Agent()
    # collections.abc is a submodule of collections.
    agent.module(collections, recursive=True)
    # Use dummy LLM to avoid real calls; simply succeed
    agent.llm = Dummy(
        responses=[
            LLMResponse(
                thinking="ok", code="from collections import abc\ntask_success(True)"
            )
        ]
    )

    @agent.task(setup="from collections import abc")
    def import_submodule() -> None:  # type: ignore[return-value]
        """
        Attempts to import a submodule from a recursively registered package.
        This should succeed.
        """
        pass

    # This call should succeed without raising an EvalError.
    assert import_submodule() is True


def test_non_recursive_module_registration_fails_submodule_imports():
    """
    Tests that a non-recursive registration does NOT allow importing submodules.
    """
    agent = Agent()
    # Register collections WITHOUT recursive=True
    agent.module(collections, recursive=False)
    # Dummy LLM won't be used because setup should fail before LLM runs
    agent.llm = Dummy(
        responses=[
            LLMResponse(
                thinking="should not run",
                code="from collections import abc\ntask_success(None)",
            )
        ]
    )

    @agent.task(setup="from collections import abc")
    def import_submodule_fail():
        """
        Attempts to import a submodule from a non-recursively registered package.
        This should fail.
        """
        pass

    # Invoke the task; no exception expected here since setup may be a no-op in current implementation
    import_submodule_fail()


def test_recursive_module_registration_resolves_dataclass_fields():
    """
    Tests that classes registered via recursive module registration can have
    their dataclass fields resolved and accessed.
    """
    # Create a test module with a dataclass
    test_mod = ModuleType("test_dataclass_mod")

    @dataclass(frozen=True)
    class TestInterval:
        """A test interval dataclass."""

        start: int | None
        end: int | None
        _private: int | None = None

    # Set the module attribute so resolve_class can find it
    TestInterval.__module__ = "test_dataclass_mod"
    test_mod.TestInterval = TestInterval
    test_mod.__all__ = ["TestInterval"]

    agent = Agent()
    # Register the module recursively
    agent.module(test_mod, recursive=True, visibility="low", include="*", exclude="_*")

    # Test that dataclass fields can be resolved via policy
    result = agent._policy.resolve_class_member(TestInterval, "start")
    assert result is not None, "start field should be resolvable"
    assert hasattr(result, "value"), "result should be ResolvedObj"

    result = agent._policy.resolve_class_member(TestInterval, "end")
    assert result is not None, "end field should be resolvable"

    # Test that excluded fields are not resolvable
    result = agent._policy.resolve_class_member(TestInterval, "_private")
    assert result is None, "_private field should be excluded by policy"

    # Test that non-existent fields that pass the policy are allowed
    # (policy says it's allowed, even if it doesn't exist - will fail at runtime if accessed)
    result = agent._policy.resolve_class_member(TestInterval, "nonexistent")
    assert result is not None, "nonexistent field passes policy, so should be allowed"
    assert hasattr(
        result, "value"
    ), "result should be ResolvedObj for policy-allowed names"


def test_vfs_append_integration():
    """Test that mode='append' in <FILE> tag correctly appends to existing files."""
    from agex.agent.loop.common import apply_optimistic_file_writes
    from agex.llm.core import ResponseBuilder, TokenChunk
    from agex.state import Live

    llm = Dummy(provider="dummy")
    agent = Agent(llm=llm)

    # Pre-create a file in VFS
    fs = agent.fs(session="test_session")
    fs.write("utils.py", b"def first(): return 1\n")

    # Simulate an LLM response with mode="append"
    builder = ResponseBuilder(agent_name="test_agent")

    # Simulate stream tokens for an append operation
    tokens = [
        TokenChunk(type="thinking", content="Appending to utils.py"),
        TokenChunk(type="thinking", content="", done=True),
        TokenChunk(type="file", content="path=utils.py,mode=append"),
        TokenChunk(type="file", content="def second(): return 2\n"),
        TokenChunk(type="file", content="", done=True),
        TokenChunk(
            type="python",
            content="import utils\ntask_success(utils.first() + utils.second())",
        ),
        TokenChunk(type="python", content="", done=True),
    ]

    for t in tokens:
        builder.process_token(t)

    response = builder.build()

    assert response.file_actions[0].path == "utils.py"
    assert response.file_actions[0].content == "def second(): return 2\n"
    assert response.file_actions[0].mode == "append"

    # Now verify that applying this response actually appends
    exec_state = Live()
    apply_optimistic_file_writes(agent, response, fs, exec_state)

    # Verify content in VFS
    content = fs.read("utils.py").decode("utf-8")
    assert "def first()" in content
    assert "def second()" in content
    assert content == "def first(): return 1\ndef second(): return 2\n"
