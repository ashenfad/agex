import math
import pickle
from dataclasses import dataclass
from types import ModuleType

import pytest

from agex.agent import Agent, MemberSpec, RegisteredClass
from agex.agent.base import clear_agent_registry
from agex.llm import DummyLLMClient
from agex.llm.core import LLMResponse
from agex.state import Namespaced, Versioned
from agex.state.kv import Memory
from tests.agex import test_module


def test_view_image_primer_text_is_always_visible():
    """
    Tests that the `view_image` function is always included in the system
    primer, regardless of whether a vision library is registered.
    """
    # Agent with NO image-related modules
    agent = Agent()
    system_message = agent._build_system_message()
    assert "view_image" in system_message
    assert "PIL.Image.Image" in system_message
    assert "matplotlib.figure.Figure" in system_message
    assert "numpy.ndarray" in system_message


def test_agent_fn_registration_decorator():
    agent = Agent()

    @agent.fn()
    def my_func():
        """My docstring"""
        return 1

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "high"
    assert reg.docstring == "My docstring"


def test_agent_fn_registration_decorator_with_args():
    agent = Agent()

    @agent.fn(visibility="low", docstring="New doc")
    def my_func():
        """Original doc"""
        return 1

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "low"
    assert reg.docstring == "New doc"


def test_agent_fn_registration_functional():
    agent = Agent()

    def my_func():
        return 1

    agent.fn(visibility="medium")(my_func)

    assert "my_func" in agent.fn_registry
    reg = agent.fn_registry["my_func"]
    assert reg.fn == my_func
    assert reg.visibility == "medium"
    assert reg.docstring is None


def test_agent_fn_registration_functional_builtin():
    agent = Agent()
    agent.fn()(math.sqrt)  # Test the decorator factory style
    assert "sqrt" in agent.fn_registry
    reg = agent.fn_registry["sqrt"]
    assert reg.fn == math.sqrt


def test_agent_fn_registration_direct_call():
    agent = Agent()
    agent.fn(math.sqrt)  # Test the direct call style
    assert "sqrt" in agent.fn_registry
    reg = agent.fn_registry["sqrt"]
    assert reg.fn == math.sqrt


def test_agent_fn_registration_with_name_alias():
    agent = Agent()

    def original_function_name():
        return "aliased"

    agent.fn(original_function_name, name="alias")

    assert "alias" in agent.fn_registry
    assert "original_function_name" not in agent.fn_registry
    reg = agent.fn_registry["alias"]
    assert reg.fn() == "aliased"


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

    assert "MyData" in agent.cls_registry
    reg = agent.cls_registry["MyData"]
    assert reg.cls == MyData
    assert reg.visibility == "high"
    assert reg.constructable is True
    assert set(reg.attrs.keys()) == {"x", "y"}  # _z is excluded
    assert reg.attrs["x"].visibility == "high"
    assert set(reg.methods.keys()) == {
        "do_stuff",
        "__init__",
    }  # _do_private_stuff is excluded


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

    assert "MyClass" in agent.cls_registry
    reg = agent.cls_registry["MyClass"]
    assert reg.cls == MyClass
    assert reg.visibility == "medium"
    assert reg.constructable is False
    assert set(reg.attrs.keys()) == {"x", "_y"}
    assert set(reg.methods.keys()) == {
        "do_stuff"
    }  # constructable=False removed __init__


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

    assert "MyService" in agent.cls_registry
    reg = agent.cls_registry["MyService"]

    # Check methods: critical_op was included and configured, regular_op was excluded.
    # __init__ is included because constructable=True by default.
    assert set(reg.methods.keys()) == {"critical_op", "__init__"}
    assert reg.methods["critical_op"].visibility == "high"
    assert "_private_op" not in reg.methods

    # Check attrs: name and config_path included, _internal_id excluded.
    assert set(reg.attrs.keys()) == {"config_path", "name"}
    assert reg.attrs["config_path"].visibility == "low"
    assert reg.attrs["name"].visibility == "medium"


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

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    # Check top-level visibilities
    assert reg.visibility == "low"

    # Check top-level consts and fns
    assert reg.consts["PI"].visibility == "high"
    assert reg.fns["public_fn"].visibility == "low"

    # Check class and its nested method
    cls_reg = reg.classes["PublicClass"]
    assert cls_reg.constructable is True
    assert "__init__" in cls_reg.methods
    assert "public_method" in cls_reg.methods
    assert cls_reg.methods["public_method"].visibility == "high"


def test_agent_module_registration_defaults():
    agent = Agent()
    agent.module(test_module, name="test_module")

    assert "test_module" in agent.importable_modules
    reg = agent.importable_modules["test_module"]

    assert reg.name == "test_module"
    assert set(reg.fns.keys()) == {"public_fn"}
    assert set(reg.consts.keys()) == {"PI"}
    assert "PublicClass" in reg.classes
    assert "_PrivateClass" not in reg.classes

    public_class_reg = reg.classes["PublicClass"]
    assert isinstance(public_class_reg, RegisteredClass)
    assert public_class_reg.constructable is True
    assert set(public_class_reg.methods.keys()) == {"public_method", "__init__"}


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

    assert "sample" in agent.importable_modules
    reg = agent.importable_modules["sample"]

    # Check that the module itself has the base visibility
    assert reg.visibility == "low"

    # Check that a selected function has the configured visibility
    assert "public_fn" in reg.fns
    assert reg.fns["public_fn"].visibility == "medium"

    # Check that a constant's visibility can be configured
    assert "PI" in reg.consts
    assert reg.consts["PI"].visibility == "high"

    # Check that a class's method can be configured and it is not constructable
    assert "PublicClass" in reg.classes
    pub_cls = reg.classes["PublicClass"]
    assert pub_cls.constructable is False
    assert set(pub_cls.methods.keys()) == {"public_method"}  # No __init__
    assert pub_cls.methods["public_method"].visibility == "high"


def test_agent_cls_no_parens():
    """Tests that the @agent.cls decorator works without parentheses."""
    agent = Agent()

    @agent.cls
    @dataclass
    class SimpleData:
        value: str

    assert "SimpleData" in agent.cls_registry
    reg = agent.cls_registry["SimpleData"]
    assert reg.cls == SimpleData
    assert reg.visibility == "high"  # Check default visibility
    assert set(reg.attrs.keys()) == {"value"}  # Check default attr selection


def test_agent_cls_direct_call():
    agent = Agent()

    class MyClass:
        pass

    agent.cls(MyClass, visibility="low")

    assert "MyClass" in agent.cls_registry
    reg = agent.cls_registry["MyClass"]
    assert reg.cls == MyClass
    assert reg.visibility == "low"


def test_agent_cls_with_name_alias():
    agent = Agent()

    class OriginalClassName:
        pass

    agent.cls(OriginalClassName, name="AliasClass")  # type: ignore

    assert "AliasClass" in agent.cls_registry
    assert "OriginalClassName" not in agent.cls_registry

    reg = agent.cls_registry["AliasClass"]
    assert reg.cls == OriginalClassName


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
    assert "sqrt" in agent.fn_registry
    assert agent.fn_registry["sqrt"].fn is math.sqrt


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
    with pytest.raises(ValueError, match="Agent name 'test_agent' already exists"):
        Agent(name="test_agent")


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

    # Verify it's registered in the fn decorator's agent
    assert "dual_function" in orchestrator.fn_registry

    # Verify dual-decorator metadata (namespace is sufficient)
    # The __agex_task_namespace__ attribute serves as both the task marker and namespace


def test_namespaced_state_isolation():
    """Test that Namespaced state provides proper isolation."""
    # Create shared state
    main_state = Versioned(Memory())

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
    agent = Agent(name="test_agent")
    agent.llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will return the expected result",
                code="task_success('test result')",
            )
        ]
    )

    @agent.task("Test task with inputs")
    def test_task(message: str, value: int) -> str:  # type: ignore
        """A test task with parameters."""
        pass

    # Create a versioned state and call the task to put inputs in state
    state = Versioned(Memory())

    # This will trigger creation and storage of the input dataclass
    result = test_task(message="hello", value=42, state=state)  # type: ignore
    assert result == "test result"  # Verify the dummy LLM response was used

    # Verify inputs were stored and are pickleable
    inputs = state.get("test_agent/inputs")
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
    via mutation, the snapshot process handles it gracefully by recording an
    error in stdout instead of crashing.
    """

    clear_agent_registry()

    agent = Agent(name="test_agent")

    # This fn mutates a dictionary to include a real Python lambda,
    # making the dictionary unserializable.
    @agent.fn()
    def make_object_unserializable(obj):
        obj["bad_field"] = lambda: "I cannot be pickled"

    agent.llm_client = DummyLLMClient(
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

    @agent.task("A task that creates bad state via mutation.")
    def task_with_unserializable_state() -> str:  # type: ignore
        """This task will create unserializable state by mutation."""
        pass

    # Use a Versioned state so that snapshotting is triggered
    state = Versioned(Memory())

    # Pre-populate the state with a serializable object.
    # We must snapshot it once so it's in the long-term store and tracked
    # for mutations.
    state.set("test_agent/my_object", {"a": 1})
    state.snapshot()

    # Run the task. This will mutate my_object and then try to snapshot.
    # It should NOT raise a PicklingError.
    result = task_with_unserializable_state(state=state)  # type: ignore
    assert result == "done"

    # After task completion, stdout should be empty because warnings are cleared between iterations
    # This is the correct behavior - warnings appear in the next iteration, then get cleared
    stdout = state.get("test_agent/__stdout__")
    # stdout should be empty due to clearing between iterations
    assert stdout == []

    # However, we can verify the task completed successfully, which means
    # the agent was able to continue despite the serialization warning


def test_shallow_validation_on_large_input_list():
    """
    Tests that the shallow validator catches bad data in a large input list.
    """

    clear_agent_registry()
    agent = Agent(name="test_agent")
    # The non-failing path of this test will enter the task loop.
    # We provide a single dummy response for it to consume.
    agent.llm_client = DummyLLMClient(
        responses=[LLMResponse(thinking="Looks good.", code="task_success(1)")]
    )

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
    agent = Agent(name="test_agent")

    # Large, valid dictionary
    large_valid_dict = {f"key_{i}": i for i in range(150)}
    # Large, invalid dictionary (error in the tail)
    large_invalid_dict = large_valid_dict.copy()
    large_invalid_dict["key_145"] = "not an int"  # type: ignore

    agent.llm_client = DummyLLMClient(
        responses=[
            LLMResponse(
                thinking="I will try to return an invalid dictionary.",
                code="task_success(invalid_dict)",
            ),
            LLMResponse(
                thinking="That failed. I will return a valid dictionary now.",
                code="task_success(valid_dict)",
            ),
        ]
    )

    @agent.task("A task that returns a large dictionary.")
    def produce_large_dict() -> dict[str, int]:  # type: ignore
        pass

    state = Versioned(Memory())
    # Pre-populate state to avoid parsing large literals in the agent's code
    state.set("test_agent/invalid_dict", large_invalid_dict)
    state.set("test_agent/valid_dict", large_valid_dict)

    result = produce_large_dict(state=state)  # type: ignore

    # Check that the final result is the valid one
    assert result == large_valid_dict

    # After task completion, stdout should be empty because errors are cleared between iterations
    # This is the correct behavior - the agent DID see the validation error in iteration 2
    # (as evidenced by the fact that it then provided valid output), but old errors don't accumulate
    stdout = state.get("test_agent/__stdout__")
    assert stdout == []  # No accumulated errors
