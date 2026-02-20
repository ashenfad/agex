"""
Tests for the sblite bridge layer.

Verifies that the bridge correctly translates agex policy → sblite policy,
builds namespaces, handles results, and executes code through sblite's sandbox.
"""

import math

import pytest
from kvit import Live
from sblite import ExecResult

from agex.agent import Agent, clear_agent_registry
from agex.agent.datatypes import TaskFail, TaskSuccess
from agex.agent.events import OutputEvent
from agex.eval.bridge import execute_sandboxed
from agex.eval.bridge.namespace import build_namespace
from agex.eval.bridge.policy import translate_policy
from agex.eval.bridge.result import handle_result
from agex.state import events


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

    def test_hydrates_state(self):
        state = Live()
        state.set("x", 42)
        state.set("name", "alice")
        agent = Agent(name="ns_test")
        ns, pre_keys = build_namespace(state, agent, "ns_test")
        assert ns["x"] == 42
        assert ns["name"] == "alice"
        assert pre_keys == {"x", "name"}

    def test_skips_internal_keys(self):
        state = Live()
        state.set("__event_log__", [])
        state.set("__expected_return_type__", str)
        state.set("x", 1)
        agent = Agent(name="ns_test")
        ns, pre_keys = build_namespace(state, agent, "ns_test")
        assert "__event_log__" not in ns
        assert "__expected_return_type__" not in ns
        assert "x" in ns
        assert pre_keys == {"x"}

    def test_task_control_present(self):
        state = Live()
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(state, agent, "ns_test")
        assert callable(ns["task_success"])
        assert callable(ns["task_fail"])
        assert callable(ns["task_clarify"])
        assert callable(ns["task_continue"])

    def test_stateful_builtins_present(self):
        state = Live()
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(state, agent, "ns_test")
        # print is handled via Sandbox print_handler, not in namespace
        assert "print" not in ns
        assert callable(ns["view_image"])
        assert callable(ns["help"])
        assert callable(ns["dir"])

    def test_print_handler_creates_output_event(self):
        """Test that make_print_handler captures real objects into OutputEvent."""
        from agex.eval.bridge.namespace import make_print_handler

        state = Live()
        state.set("__event_log__", [])
        handler = make_print_handler(state, "test_agent", None)

        handler("Hello", 42)

        event_list = events(state)
        assert len(event_list) == 1
        assert isinstance(event_list[0], OutputEvent)
        assert event_list[0].parts == ["Hello", 42]

    def test_task_continue_raises(self):
        from agex.agent.datatypes import TaskContinue

        state = Live()
        state.set("__event_log__", [])
        agent = Agent(name="ns_test")
        ns, _ = build_namespace(state, agent, "ns_test")

        with pytest.raises(TaskContinue):
            ns["task_continue"]()


class TestResultHandler:
    """Tests for handle_result()."""

    def test_syncs_new_values(self):
        state = Live()
        result = ExecResult(namespace={"x": 42, "y": "hello"})
        handle_result(result, state, "test", set())
        assert state.get("x") == 42
        assert state.get("y") == "hello"

    def test_skips_internal_keys(self):
        state = Live()
        result = ExecResult(namespace={"x": 1, "__builtins__": {}})
        handle_result(result, state, "test", set())
        assert state.get("x") == 1
        assert "__builtins__" not in state

    def test_detects_deletions(self):
        state = Live()
        state.set("x", 42)
        state.set("y", "keep")
        pre_keys = {"x", "y"}
        # Only y remains in namespace after exec
        result = ExecResult(namespace={"y": "keep"})
        handle_result(result, state, "test", pre_keys)
        assert "x" not in state
        assert state.get("y") == "keep"

    def test_reraises_task_success(self):
        state = Live()
        result = ExecResult(error=TaskSuccess("done"))
        with pytest.raises(TaskSuccess):
            handle_result(result, state, "test", set())

    def test_reraises_task_fail(self):
        state = Live()
        result = ExecResult(error=TaskFail("oops"))
        with pytest.raises(TaskFail):
            handle_result(result, state, "test", set())

    def test_reraises_regular_exception(self):
        state = Live()
        result = ExecResult(error=ValueError("bad value"))
        with pytest.raises(ValueError, match="bad value"):
            handle_result(result, state, "test", set())

    def test_syncs_before_reraising(self):
        """State should be synced even when an error is re-raised."""
        state = Live()
        result = ExecResult(
            namespace={"x": 42},
            error=TaskSuccess("done"),
        )
        with pytest.raises(TaskSuccess):
            handle_result(result, state, "test", set())
        # x should still be synced despite the error
        assert state.get("x") == 42


class TestExecuteSandboxed:
    """Integration tests for execute_sandboxed()."""

    def setup_method(self):
        clear_agent_registry()

    def test_simple_assignment(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("x = 42", agent, state)
        assert state.get("x") == 42

    def test_arithmetic(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("result = 2 + 3 * 4", agent, state)
        assert state.get("result") == 14

    def test_task_success_raises(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        with pytest.raises(TaskSuccess) as exc_info:
            execute_sandboxed('task_success("done")', agent, state)
        assert exc_info.value.result == "done"

    def test_task_fail_raises(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        with pytest.raises(TaskFail):
            execute_sandboxed('task_fail("oops")', agent, state)

    def test_print_creates_event(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed('print("Hello from sandbox")', agent, state)
        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].parts == ["Hello from sandbox"]

    def test_state_persistence_across_calls(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("x = 10", agent, state)
        execute_sandboxed("y = x + 5", agent, state)
        assert state.get("y") == 15

    def test_variable_deletion(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        state.set("x", 42)
        execute_sandboxed("del x", agent, state)
        assert "x" not in state

    def test_registered_module_import(self):
        agent = Agent(name="exec_test")
        agent.module(math)
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("import math\nresult = math.sqrt(16)", agent, state)
        assert state.get("result") == 4.0

    def test_registered_function_call(self):
        agent = Agent(name="exec_test")

        def double(n):
            return n * 2

        agent.fn(double)
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("result = double(21)", agent, state)
        assert state.get("result") == 42

    def test_registered_class_instantiation(self):
        agent = Agent(name="exec_test")

        class Point:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        agent.cls(Point)
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("p = Point(3, 4)", agent, state)
        p = state.get("p")
        assert p.x == 3
        assert p.y == 4

    def test_exception_propagates(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        with pytest.raises(ZeroDivisionError):
            execute_sandboxed("x = 1 / 0", agent, state)

    def test_syntax_error(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        with pytest.raises(SyntaxError):
            execute_sandboxed("def f(:", agent, state)

    def test_user_defined_function(self):
        agent = Agent(name="exec_test")
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed(
            "def square(n):\n    return n * n\nresult = square(7)",
            agent,
            state,
        )
        assert state.get("result") == 49

    def test_from_import(self):
        agent = Agent(name="exec_test")
        agent.module(math)
        state = Live()
        state.set("__event_log__", [])
        execute_sandboxed("from math import pi\nresult = round(pi, 2)", agent, state)
        assert state.get("result") == 3.14
