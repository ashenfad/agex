"""
Tests for the sandtrap bridge layer.

Verifies that the bridge correctly translates agex policy → sandtrap policy,
builds namespaces, handles results, and executes code through sandtrap's sandbox.
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

    def test_hydrates_state(self):
        state = Live()
        state["x"] = 42
        state["name"] = "alice"
        agent = Agent(name="ns_test")
        ns, pre_keys, _ = build_namespace(state, agent, "ns_test")
        assert ns["x"] == 42
        assert ns["name"] == "alice"
        assert pre_keys == {"x", "name"}

    def test_skips_internal_keys(self):
        state = Live()
        state["__event_log__"] = []
        state["__expected_return_type__"] = str
        state["x"] = 1
        agent = Agent(name="ns_test")
        ns, pre_keys, _ = build_namespace(state, agent, "ns_test")
        assert "__event_log__" not in ns
        assert "__expected_return_type__" not in ns
        assert "x" in ns
        assert pre_keys == {"x"}

    def test_task_control_present(self):
        state = Live()
        agent = Agent(name="ns_test")
        ns, _, _ = build_namespace(state, agent, "ns_test")
        assert callable(ns["task_success"])
        assert callable(ns["task_fail"])
        assert callable(ns["task_clarify"])
        assert callable(ns["task_continue"])

    def test_view_image_present(self):
        state = Live()
        agent = Agent(name="ns_test")
        ns, _, _ = build_namespace(state, agent, "ns_test")
        # print is handled via sandtrap's snapshot_prints, not in namespace
        assert "print" not in ns
        assert callable(ns["view_image"])
        # help uses Python builtins; dir is overridden to filter internals
        assert "help" not in ns
        assert callable(ns["dir"])

    def test_outputs_list_present(self):
        state = Live()
        agent = Agent(name="ns_test")
        ns, _, _ = build_namespace(state, agent, "ns_test")
        assert isinstance(ns["__outputs__"], list)
        assert len(ns["__outputs__"]) == 0

    def test_print_snapshot_creates_output_event(self):
        """Test that result.prints are converted to OutputEvents by handle_result."""
        state = Live()
        state["__event_log__"] = []
        result = ExecResult(prints=[("Hello", 42)])

        handle_result(result, state, "test_agent", set())

        event_list = events(state)
        assert len(event_list) == 1
        assert isinstance(event_list[0], OutputEvent)
        assert event_list[0].parts == ["Hello", 42]

    def test_task_continue_raises(self):
        from agex.agent.datatypes import TaskContinue

        state = Live()
        state["__event_log__"] = []
        agent = Agent(name="ns_test")
        ns, _, _ = build_namespace(state, agent, "ns_test")

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
        state["x"] = 42
        state["y"] = "keep"
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


class TestChangeDetection:
    """Tests for identity-based change detection in handle_result.

    Only variables that were reassigned (different id()) or newly created
    should be written to state.  Unchanged variables should be skipped to
    avoid creating duplicate blobs in kvgit.
    """

    def test_unchanged_variable_not_restaged(self):
        """A variable with the same id() after execution should not be
        written back to state (no unnecessary staging)."""
        state = Live()
        original = [1, 2, 3]
        state["data"] = original

        # Simulate: namespace has the same object (not reassigned)
        ns_obj = original  # same id()
        pre_ids = {"data": id(ns_obj)}
        result = ExecResult(namespace={"data": ns_obj})

        handle_result(result, state, "test", {"data"}, pre_ids=pre_ids)
        # state["data"] should not have been re-set (no __setitem__ call)
        # We verify by checking that the Live store wasn't written to —
        # if it were, .get() would return a re-pickled copy, not the original.
        # For Live(), __setitem__ always writes, so the test is that the
        # value is correct (and in production with Staged, the key wouldn't
        # be in _updates).
        assert state.get("data") == [1, 2, 3]

    def test_reassigned_variable_is_staged(self):
        """A variable with a different id() should be written to state."""
        state = Live()
        state["x"] = 10

        old_obj = state.get("x")  # the hydrated value
        new_obj = 20  # different id
        pre_ids = {"x": id(old_obj)}
        result = ExecResult(namespace={"x": new_obj})

        handle_result(result, state, "test", {"x"}, pre_ids=pre_ids)
        assert state.get("x") == 20

    def test_new_variable_is_staged(self):
        """A variable not in pre_ids (newly created) should be written."""
        state = Live()
        pre_ids = {}  # nothing existed before
        result = ExecResult(namespace={"new_var": 42})

        handle_result(result, state, "test", set(), pre_ids=pre_ids)
        assert state.get("new_var") == 42

    def test_deletion_still_works_with_pre_ids(self):
        """Variable deletion should still work when pre_ids is provided."""
        state = Live()
        state["x"] = 42
        state["y"] = 99

        pre_ids = {"x": id(42), "y": id(99)}
        result = ExecResult(namespace={"y": 99})  # x deleted

        handle_result(result, state, "test", {"x", "y"}, pre_ids=pre_ids)
        assert "x" not in state
        assert state.get("y") == 99

    def test_without_pre_ids_all_written(self):
        """Without pre_ids (backward compat), all variables are written."""
        state = Live()
        state["x"] = 10
        result = ExecResult(namespace={"x": 10, "y": 20})

        handle_result(result, state, "test", {"x"})
        assert state.get("x") == 10
        assert state.get("y") == 20

    def test_mixed_changed_and_unchanged(self):
        """Only changed/new variables are staged; unchanged are skipped."""
        state = Live()
        big_data = {"key": "value" * 1000}
        state["big"] = big_data
        state["small"] = 1

        # Simulate: big is same object, small is reassigned
        pre_ids = {"big": id(big_data), "small": id(state.get("small"))}
        result = ExecResult(
            namespace={
                "big": big_data,  # same id — should skip
                "small": 2,  # different value, different id
                "new": "hello",  # not in pre_ids — should write
            }
        )

        # Use a tracking wrapper to count writes
        writes = []

        class TrackingState:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __setitem__(self, key, value):
                writes.append(key)
                self._inner[key] = value

            def __contains__(self, key):
                return key in self._inner

            def __delitem__(self, key):
                del self._inner[key]

            def get(self, key, default=None):
                return self._inner.get(key, default)

            def keys(self):
                return self._inner.keys()

        tracking = TrackingState(state)
        handle_result(result, tracking, "test", {"big", "small"}, pre_ids=pre_ids)

        assert "big" not in writes, "Unchanged variable should not be written"
        assert "small" in writes, "Reassigned variable should be written"
        assert "new" in writes, "New variable should be written"

    def test_execute_sandboxed_skips_untouched_vars(self):
        """End-to-end: execute_sandboxed only writes variables that changed."""
        clear_agent_registry()
        agent = Agent(name="cd_test")
        state = Live()
        state["__event_log__"] = []

        # Turn 1: create two variables
        execute_sandboxed("big = list(range(1000))\nsmall = 1", agent, state)
        assert state.get("big") == list(range(1000))
        assert state.get("small") == 1

        # Wrap state to track writes on turn 2
        writes = []

        class TrackingLive:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def __setitem__(self, key, value):
                writes.append(key)
                self._inner[key] = value

            def __contains__(self, key):
                return key in self._inner

            def __delitem__(self, key):
                del self._inner[key]

            def get(self, key, default=None):
                return self._inner.get(key, default)

            def keys(self):
                return self._inner.keys()

        tracking = TrackingLive(state)

        # Turn 2: only reassign small, don't touch big
        execute_sandboxed("small = 2", agent, tracking)

        assert "small" in writes, "Reassigned variable should be written"
        assert "big" not in writes, "Untouched variable should NOT be written"


class TestTaskControlPicklability:
    """Tests that task control functions survive pickle roundtrip."""

    def test_task_control_functions_are_picklable(self):
        import pickle

        from agex.eval.bridge.namespace import (
            _task_clarify,
            _task_continue,
            _task_fail,
            _task_success,
        )

        for fn in [_task_success, _task_fail, _task_clarify, _task_continue]:
            roundtripped = pickle.loads(pickle.dumps(fn))
            assert callable(roundtripped)

    def test_task_success_validation_in_handle_result(self):
        """Validation of TaskSuccess result happens in handle_result, not in sandbox."""
        state = Live()
        state["__event_log__"] = []
        state["__expected_return_type__"] = int

        # Valid result — should re-raise TaskSuccess without TypeError
        result = ExecResult(error=TaskSuccess(result=42))
        with pytest.raises(TaskSuccess):
            handle_result(result, state, "test", set())

        # Invalid result — should raise TypeError from validation
        result = ExecResult(error=TaskSuccess(result="not an int"))
        with pytest.raises(TypeError, match="Output validation failed"):
            handle_result(result, state, "test", set())

    def test_task_continue_observations_create_event(self):
        """task_continue observations are converted to OutputEvent in handle_result."""
        from agex.agent.datatypes import TaskContinue

        state = Live()
        state["__event_log__"] = []

        result = ExecResult(error=TaskContinue(observations=("progress", 50)))
        with pytest.raises(TaskContinue):
            handle_result(result, state, "test_agent", set())

        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].parts == ["progress", 50]

    def test_task_continue_no_observations_no_event(self):
        """task_continue without observations creates no OutputEvent."""
        from agex.agent.datatypes import TaskContinue

        state = Live()
        state["__event_log__"] = []

        result = ExecResult(error=TaskContinue())
        with pytest.raises(TaskContinue):
            handle_result(result, state, "test_agent", set())

        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 0


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

    def test_outputs_converted_to_events_by_handle_result(self):
        from agex.eval.objects import ImageAction

        state = Live()
        state["__event_log__"] = []
        img_action = ImageAction(image="test_img", detail="high")
        result = ExecResult(namespace={"__outputs__": [img_action]})

        handle_result(result, state, "test_agent", set())

        event_list = events(state)
        assert len(event_list) == 1
        assert isinstance(event_list[0], OutputEvent)
        assert len(event_list[0].parts) == 1
        assert isinstance(event_list[0].parts[0], ImageAction)
        assert event_list[0].parts[0].image == "test_img"

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
    """Integration tests for execute_sandboxed()."""

    def setup_method(self):
        clear_agent_registry()

    def test_simple_assignment(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("x = 42", agent, state)
        assert state.get("x") == 42

    def test_arithmetic(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("result = 2 + 3 * 4", agent, state)
        assert state.get("result") == 14

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
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed('print("Hello from sandbox")', agent, state)
        event_list = events(state)
        output_events = [e for e in event_list if isinstance(e, OutputEvent)]
        assert len(output_events) == 1
        assert output_events[0].parts == ["Hello from sandbox"]

    def test_state_persistence_across_calls(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("x = 10", agent, state)
        execute_sandboxed("y = x + 5", agent, state)
        assert state.get("y") == 15

    def test_variable_deletion(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
        state["x"] = 42
        execute_sandboxed("del x", agent, state)
        assert "x" not in state

    def test_registered_module_import(self):
        agent = Agent(name="exec_test")
        agent.module(math)
        state = Live()
        state["__event_log__"] = []
        execute_sandboxed("import math\nresult = math.sqrt(16)", agent, state)
        assert state.get("result") == 4.0

    def test_registered_function_call(self):
        agent = Agent(name="exec_test")

        def double(n):
            return n * 2

        agent.fn(double)
        state = Live()
        state["__event_log__"] = []
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
        state["__event_log__"] = []
        execute_sandboxed("p = Point(3, 4)", agent, state)
        p = state.get("p")
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

    def test_user_defined_function(self):
        agent = Agent(name="exec_test")
        state = Live()
        state["__event_log__"] = []
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
        state["__event_log__"] = []
        execute_sandboxed("from math import pi\nresult = round(pi, 2)", agent, state)
        assert state.get("result") == 3.14
