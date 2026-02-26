"""
End-to-end integration tests for the isolation parameter.

Verifies that the full task loop (Agent → LLM → sandbox → result) works
correctly with each isolation level, using Dummy LLMs.
"""

import sys

import pytest

from agex import Agent, TaskFail, clear_agent_registry
from agex.agent.events import OutputEvent, SuccessEvent
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy
from agex.state import connect_state, events

# Process isolation requires fork-capable multiprocessing
_skip_process = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Process isolation requires fork (Unix only)",
)


class TestIsolationNone:
    """Baseline: isolation='none' (in-process) through the full task loop."""

    def setup_method(self):
        clear_agent_registry()

    def test_task_success(self):
        llm = Dummy([LLMResponse(thinking="done", code='task_success("hello")')])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_ok", llm=llm, state=config, isolation="none")

        @agent.task
        def greet():
            """Greet."""
            pass

        result = greet(session="s")
        assert result == "hello"

    def test_registered_function(self):
        llm = Dummy([LLMResponse(thinking="call it", code="task_success(double(21))")])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_fn", llm=llm, state=config, isolation="none")

        @agent.fn()
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        @agent.task
        def calc() -> int:
            """Calculate."""
            pass

        assert calc(session="s") == 42

    def test_print_creates_output_event(self):
        llm = Dummy(
            [
                LLMResponse(
                    thinking="print then succeed",
                    code='print("hi")\ntask_success("ok")',
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_print", llm=llm, state=config, isolation="none")

        @agent.task
        def chatty():
            """Chatty task."""
            pass

        chatty(session="s")
        state = agent._host.resolve_state(config, "s")
        output = [
            e
            for e in events(state)
            if isinstance(e, OutputEvent) and e.agent_name == "none_print"
        ]
        assert len(output) >= 1
        assert output[0].parts == ["hi"]


@_skip_process
class TestIsolationProcess:
    """isolation='process' — full task loop through subprocess sandbox."""

    def setup_method(self):
        clear_agent_registry()

    def test_task_success(self):
        """task_success travels across the process boundary."""
        llm = Dummy(
            [LLMResponse(thinking="done", code='task_success("cross-process")')]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_ok",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def greet():
            """Greet."""
            pass

        result = greet(session="s")
        assert result == "cross-process"

    def test_task_success_with_events(self):
        """Verify the full event chain: TaskStart → Action → Success."""
        llm = Dummy([LLMResponse(thinking="solve", code="task_success(7 * 6)")])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_events",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def math_task() -> int:
            """Do math."""
            pass

        result = math_task(session="s")
        assert result == 42

        state = agent._host.resolve_state(config, "s")
        event_list = [e for e in events(state) if e.full_namespace == "proc_events"]
        success = [e for e in event_list if isinstance(e, SuccessEvent)]
        assert len(success) == 1
        assert success[0].result == 42

    def test_task_fail(self):
        """task_fail travels across the process boundary."""
        llm = Dummy([LLMResponse(thinking="fail", code='task_fail("cannot proceed")')])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_fail",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def fail_task():
            """Fail."""
            pass

        with pytest.raises(TaskFail) as exc_info:
            fail_task(session="s")
        assert exc_info.value.message == "cannot proceed"

    def test_print_creates_output_event(self):
        """print() snapshots travel across the process boundary via result.prints."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="print and succeed",
                    code='print("from subprocess")\ntask_success("done")',
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_print",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def chatty():
            """Chatty task."""
            pass

        chatty(session="s")
        state = agent._host.resolve_state(config, "s")
        output = [
            e
            for e in events(state)
            if isinstance(e, OutputEvent) and e.agent_name == "proc_print"
        ]
        assert len(output) >= 1
        assert output[0].parts == ["from subprocess"]

    def test_registered_function(self):
        """Registered functions survive pickle and work cross-process."""
        llm = Dummy([LLMResponse(thinking="call it", code="task_success(double(21))")])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_fn",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.fn()
        def double(n: int) -> int:
            """Double a number."""
            return n * 2

        @agent.task
        def calc() -> int:
            """Calculate."""
            pass

        assert calc(session="s") == 42

    def test_multi_iteration_with_continue(self):
        """task_continue works cross-process for multi-turn tasks."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="step 1",
                    code='x = 10\ntask_continue("computed x")',
                ),
                LLMResponse(
                    thinking="step 2",
                    code="task_success(x * 2)",
                ),
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_cont",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def multi_step() -> int:
            """Multi-step task."""
            pass

        result = multi_step(session="s")
        assert result == 20

    def test_state_syncs_back(self):
        """Namespace changes in the subprocess sync back to state."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="set a value",
                    code='answer = 42\ntask_success("ok")',
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_state",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def set_state():
            """Set state."""
            pass

        set_state(session="s")
        state = agent._host.resolve_state(config, "s")
        assert state.get("answer") == 42

    def test_view_image_cross_process(self):
        """view_image works cross-process via __outputs__."""
        llm = Dummy(
            [
                LLMResponse(
                    thinking="view then succeed",
                    code='view_image("test_img", detail="low")\ntask_continue()',
                ),
                LLMResponse(
                    thinking="done",
                    code='task_success("ok")',
                ),
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_vi",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def view_task():
            """View an image."""
            pass

        result = view_task(session="s")
        assert result == "ok"

        state = agent._host.resolve_state(config, "s")
        output = [
            e
            for e in events(state)
            if isinstance(e, OutputEvent) and e.agent_name == "proc_vi"
        ]
        # Should have at least one OutputEvent from view_image
        image_outputs = [e for e in output if e.parts and hasattr(e.parts[0], "image")]
        assert len(image_outputs) >= 1
        assert image_outputs[0].parts[0].image == "test_img"
        assert image_outputs[0].parts[0].detail == "low"
