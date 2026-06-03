"""
Tests for the sandtrap bridge layer.

Verifies that the bridge correctly translates agex policy → sandtrap policy,
builds a fresh namespace per emission, processes results, and executes code
through sandtrap's sandbox.

Contract: each ``python_action`` runs as a fresh script.  No state hydrates
into the namespace at the start of execution; no state is written back from
the namespace at the end.  Persistent communication is via the event log
(prints, view_image, errors) and via task terminators that surface as
exceptions.
"""

import math

import pytest
from sandtrap import ExecResult

from agex.agent import Agent, clear_agent_registry
from agex.agent.datatypes import TaskFail, TaskSuccess
from agex.agent.events import OutputEvent
from agex.eval.bridge import execute_sandboxed
from agex.eval.bridge.namespace import build_namespace
from agex.eval.bridge.policy import translate_policy
from agex.eval.bridge.result import handle_result
from agex.state import events
from agex.state.live import Live


class TestPolicyTranslation:
    """Tests for translate_policy()."""

    def setup_method(self):
        clear_agent_registry()

    def test_empty_agent(self):
        agent = Agent(name="empty")
        policy = translate_policy(agent)
        assert len(policy.functions) == 0
        assert len(policy.classes) == 0
        assert len(policy.modules) == 0

    def test_registered_function(self):
        agent = Agent(name="fn_test")

        def greet(name):
            return f"Hello, {name}!"

        agent.fn(greet)
        policy = translate_policy(agent)
        assert "greet" in policy.functions
        assert policy.functions["greet"].func is greet

    def test_registered_function_with_permissions(self):
        agent = Agent(name="perm_test")

        def read_file(path):
            pass

        agent.fn(read_file, host_fs_access=True, network_access=True)
        policy = translate_policy(agent)
        reg = policy.functions["read_file"]
        assert reg.host_fs_access is True
        assert reg.network_access is True

    def test_registered_module(self):
        agent = Agent(name="mod_test")
        agent.module(math)
        policy = translate_policy(agent)
        assert "math" in policy.modules
        assert policy.modules["math"].obj is math

    def test_registered_module_with_filters(self):
        agent = Agent(name="filter_test")
        agent.module(math, include="sqrt", exclude="_*")
        policy = translate_policy(agent)
        reg = policy.modules["math"]
        assert reg.include == "sqrt"

    def test_registered_class(self):
        agent = Agent(name="cls_test")

        class MyClass:
            def method(self):
                return 42

        agent.cls(MyClass)
        policy = translate_policy(agent)
        assert "MyClass" in policy.classes
        assert policy.classes["MyClass"].cls is MyClass

    def test_registered_instance(self):
        agent = Agent(name="inst_test")

        class DB:
            def query(self, sql):
                return []

        db = DB()
        agent.module(db, name="db")
        policy = translate_policy(agent)
        assert "db" in policy.modules
        assert policy.modules["db"].obj is db

    def test_timeout_passthrough(self):
        agent = Agent(name="timeout_test")
        policy = translate_policy(agent, timeout=42.0)
        assert policy.timeout == 42.0

    def test_timeout_defaults_to_agent(self):
        agent = Agent(name="timeout_default", eval_timeout_seconds=15.0)
        policy = translate_policy(agent)
        assert policy.timeout == 15.0


class TestNamespaceBuilder:
    """Tests for build_namespace()."""

    def setup_method(self):
        clear_agent_registry()

    def test_namespace_is_fresh(self):
        """Each call returns a clean dict — no state hydration."""
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(Live(), agent, "ns_test")

        # The only top-level keys in the dict are the bridge injections
        # — task terminators, view_image, __outputs__, dir, cache, spawn.
        assert set(ns.keys()) == {
            "task_success",
            "task_fail",
            "task_clarify",
            "task_request_permission",
            "view_image",
            "__outputs__",
            "dir",
            "cache",
            "spawn",
        }

    def test_independent_namespaces(self):
        """Two calls return distinct dicts — no shared state."""
        agent = Agent(name="ns_test")
        ns1, _ = build_namespace(Live(), agent, "ns_test")
        ns2, _ = build_namespace(Live(), agent, "ns_test")
        assert ns1 is not ns2
        assert ns1["__outputs__"] is not ns2["__outputs__"]

    def test_task_control_present(self):
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(Live(), agent, "ns_test")
        assert callable(ns["task_success"])
        assert callable(ns["task_fail"])
        assert callable(ns["task_clarify"])

    def test_view_image_present(self):
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(Live(), agent, "ns_test")
        assert callable(ns["view_image"])
        assert callable(ns["dir"])

    def test_outputs_list_present(self):
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(Live(), agent, "ns_test")
        assert isinstance(ns["__outputs__"], list)
        assert len(ns["__outputs__"]) == 0

    def test_injected_keys_returned(self):
        agent = Agent(name="ns_test")
        _, injected = build_namespace(Live(), agent, "ns_test")
        assert injected == {
            "task_success",
            "task_fail",
            "task_clarify",
            "task_request_permission",
            "view_image",
            "__outputs__",
            "dir",
            "cache",
            "spawn",
        }


class TestResultHandler:
    """Tests for handle_result()."""

    def test_print_creates_output_event(self):
        from agex.eval.objects import PrintAction

        state = Live()
        state["__event_log__"] = []
        result = ExecResult(prints=[("Hello", 42)])

        handle_result(result, state, "test_agent")

        event_list = events(state)
        assert len(event_list) == 1
        assert isinstance(event_list[0], OutputEvent)
        assert event_list[0].parts == [PrintAction(args=("Hello", 42))]

    def test_view_image_creates_output_event(self):
        from agex.eval.objects import ImageAction

        state = Live()
        state["__event_log__"] = []
        img_action = ImageAction(image="test_img", detail="high")
        result = ExecResult(namespace={"__outputs__": [img_action]})

        handle_result(result, state, "test_agent")

        event_list = events(state)
        assert len(event_list) == 1
        assert isinstance(event_list[0], OutputEvent)
        assert len(event_list[0].parts) == 1
        assert isinstance(event_list[0].parts[0], ImageAction)
        assert event_list[0].parts[0].image == "test_img"

    def test_namespace_values_not_synced_to_state(self):
        """Variables in result.namespace must NOT land in state.

        This is the central B-contract assertion: the agent's namespace
        is purely turn-local and does not flow back into kvgit-backed
        state.
        """
        state = Live()
        state["__event_log__"] = []
        # Pre-existing key the agent didn't touch
        state["existing"] = "kept"

        result = ExecResult(namespace={"x": 42, "y": "hello"})
        handle_result(result, state, "test")

        assert "x" not in state
        assert "y" not in state
        assert state.get("existing") == "kept"

    def test_namespace_deletions_not_propagated(self):
        """If a state key isn't in the post-exec namespace, state must
        keep it — handle_result no longer infers deletions."""
        state = Live()
        state["__event_log__"] = []
        state["x"] = 42

        # Sandbox returns no namespace (or a different one)
        result = ExecResult(namespace={})
        handle_result(result, state, "test")

        assert state.get("x") == 42

    def test_reraises_task_success(self):
        state = Live()
        state["__event_log__"] = []
        result = ExecResult(error=TaskSuccess("done"))
        with pytest.raises(TaskSuccess):
            handle_result(result, state, "test")

    def test_reraises_task_fail(self):
        state = Live()
        state["__event_log__"] = []
        result = ExecResult(error=TaskFail("oops"))
        with pytest.raises(TaskFail):
            handle_result(result, state, "test")

    def test_reraises_regular_exception(self):
        state = Live()
        state["__event_log__"] = []
        result = ExecResult(error=ValueError("bad value"))
        with pytest.raises(ValueError, match="bad value"):
            handle_result(result, state, "test")

    def test_validates_task_success_result_type(self):
        """When a return type is registered, task_success enforces it."""
        state = Live()
        state["__event_log__"] = []
        state["__expected_return_type__"] = int

        # Valid result — should re-raise TaskSuccess without TypeError
        result = ExecResult(error=TaskSuccess(result=42))
        with pytest.raises(TaskSuccess):
            handle_result(result, state, "test")

        # Invalid result — should raise TypeError from validation
        result = ExecResult(error=TaskSuccess(result="not an int"))
        with pytest.raises(TypeError, match="Output validation failed"):
            handle_result(result, state, "test")


class TestTaskControlPicklability:
    """Tests that task control functions survive pickle roundtrip."""

    def test_task_control_functions_are_picklable(self):
        import pickle

        from agex.eval.bridge.namespace import (
            _task_clarify,
            _task_fail,
            _task_success,
        )

        for fn in [_task_success, _task_fail, _task_clarify]:
            roundtripped = pickle.loads(pickle.dumps(fn))
            assert callable(roundtripped)


class TestViewImage:
    """Tests for the __outputs__-based view_image."""

    def setup_method(self):
        clear_agent_registry()

    def test_view_image_is_picklable(self):
        import pickle

        from agex.eval.bridge.namespace import _ViewImage

        outputs = []
        vi = _ViewImage(outputs)
        roundtripped = pickle.loads(pickle.dumps(vi))
        assert callable(roundtripped)

    def test_view_image_appends_to_outputs(self):
        from agex.eval.bridge.namespace import _ViewImage
        from agex.eval.objects import ImageAction

        outputs = []
        vi = _ViewImage(outputs)
        vi("fake_image", detail="low")
        assert len(outputs) == 1
        assert isinstance(outputs[0], ImageAction)
        assert outputs[0].image == "fake_image"
        assert outputs[0].detail == "low"

    def test_view_image_invalid_detail(self):
        from agex.eval.bridge.namespace import _ViewImage

        vi = _ViewImage([])
        with pytest.raises(ValueError, match="detail must be"):
            vi("img", detail="medium")

    def test_view_image_in_sandbox(self):
        """view_image works through execute_sandboxed via __outputs__."""
        agent = Agent(name="vi_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed('view_image("test_img", detail="low")', agent, state)
        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].parts[0].image == "test_img"
        assert output_events[0].parts[0].detail == "low"


class TestExecuteSandboxed:
    """Integration tests for execute_sandboxed().

    Under the stateless contract, ``state`` is used only for the event
    log and for return-type validation — sandboxed code never reads
    user-visible variables out of state, and assignments inside a
    ``python_action`` never land in state.  These tests assert the
    observable surfaces: terminator exceptions, output events, errors.
    """

    def setup_method(self):
        clear_agent_registry()

    def test_task_success_raises(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(TaskSuccess) as exc_info:
            execute_sandboxed('task_success("done")', agent, state)
        assert exc_info.value.result == "done"

    def test_task_fail_raises(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(TaskFail):
            execute_sandboxed('task_fail("oops")', agent, state)

    def test_print_creates_event(self):
        from agex.eval.objects import PrintAction

        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed('print("Hello from sandbox")', agent, state)
        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].parts == [PrintAction(args=("Hello from sandbox",))]

    def test_assignment_does_not_leak_into_state(self):
        """The B-contract end-to-end: agent code that assigns variables
        must NOT have those assignments visible in state afterwards."""
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("x = 42", agent, state)
        assert "x" not in state

    def test_state_does_not_persist_across_calls(self):
        """Two execute_sandboxed calls must NOT share namespace state.

        Each python_action emission is a fresh script; the second call
        cannot see variables defined in the first.
        """
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("x = 10", agent, state)

        # The second emission can't see x — should raise NameError.
        with pytest.raises(NameError):
            execute_sandboxed("y = x + 5", agent, state)

    def test_registered_module_visible(self):
        """Registered modules are available without flowing through state."""
        agent = Agent(name="exec_test")
        agent.module(math)
        state = Live()
        state["__event_log__"] = []
        # Compute and print rather than asserting state.
        execute_sandboxed(
            "import math\nprint(math.sqrt(16))",
            agent,
            state,
        )
        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert any("4.0" in str(e.parts[0].args) for e in output_events)

    def test_registered_function_callable(self):
        agent = Agent(name="exec_test")

        def double(n):
            return n * 2

        agent.fn(double)
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(TaskSuccess) as exc_info:
            execute_sandboxed("task_success(double(21))", agent, state)
        assert exc_info.value.result == 42

    def test_registered_class_instantiable(self):
        agent = Agent(name="exec_test")

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        agent.cls(Point)
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(TaskSuccess) as exc_info:
            execute_sandboxed("task_success(Point(3, 4))", agent, state)
        p = exc_info.value.result
        assert p.x == 3
        assert p.y == 4

    def test_exception_propagates(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(ZeroDivisionError):
            execute_sandboxed("x = 1 / 0", agent, state)

    def test_syntax_error(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(SyntaxError):
            execute_sandboxed("def f(:", agent, state)

    def test_user_defined_function_within_emission(self):
        """A function defined inside a python_action is callable within
        the same emission (but not across emissions)."""
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        with pytest.raises(TaskSuccess) as exc_info:
            execute_sandboxed(
                "def square(n):\n    return n * n\ntask_success(square(7))",
                agent,
                state,
            )
        assert exc_info.value.result == 49
