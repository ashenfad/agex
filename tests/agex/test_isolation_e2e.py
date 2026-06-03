"""
End-to-end integration tests for the isolation parameter.

Verifies that the full task loop (Agent → LLM → sandbox → result) works
correctly with each isolation level, using Dummy LLMs.
"""

import sys

import pytest

from agex import Agent, TaskFail, clear_agent_registry
from agex.agent.datatypes import TaskTimeout
from agex.agent.events import OutputEvent, SuccessEvent
from agex.llm.dummy_client import Dummy
from agex.state import connect_state, events
from tests.agex._emissions import make_response

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
        llm = Dummy([make_response(thinking="done", code='task_success("hello")')])
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_ok", llm=llm, state=config, isolation="none")

        @agent.task
        def greet():
            """Greet."""
            pass

        result = greet(session="s")
        assert result == "hello"

    def test_registered_function(self):
        llm = Dummy(
            [make_response(thinking="call it", code="task_success(double(21))")]
        )
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

    def test_dir_returns_user_names(self):
        llm = Dummy(
            [
                make_response(
                    thinking="use dir",
                    code="x = 1\nnames = dir()\ntask_success(names)",
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_dir", llm=llm, state=config, isolation="none")

        @agent.task
        def check_dir() -> list:
            """Check dir."""
            pass

        result = check_dir(session="s")
        assert "x" in result
        assert "task_success" in result
        # Sandbox internals should be filtered
        assert not any(name.startswith("__st_") for name in result)
        assert "__builtins__" not in result

    def test_help_output_captured(self):
        llm = Dummy(
            [
                make_response(
                    thinking="use help",
                    code='help(int)\ntask_success("ok")',
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(name="none_help", llm=llm, state=config, isolation="none")

        @agent.task
        def check_help():
            """Check help."""
            pass

        result = check_help(session="s")
        assert result == "ok"
        state = agent._host.resolve_state(config, "s")
        output = [
            e
            for e in events(state)
            if isinstance(e, OutputEvent) and e.agent_name == "none_help"
        ]
        # help() output should be captured and visible
        assert len(output) >= 1
        combined = " ".join(str(p) for e in output for p in e.parts)
        assert "int" in combined

    def test_print_creates_output_event(self):
        llm = Dummy(
            [
                make_response(
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
        assert output[0].parts[0].args == ("hi",)


@_skip_process
class TestIsolationProcess:
    """isolation='process' — full task loop through subprocess sandbox."""

    def setup_method(self):
        clear_agent_registry()

    def test_task_success(self):
        """task_success travels across the process boundary."""
        llm = Dummy(
            [make_response(thinking="done", code='task_success("cross-process")')]
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
        llm = Dummy([make_response(thinking="solve", code="task_success(7 * 6)")])
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
        llm = Dummy(
            [make_response(thinking="fail", code='task_fail("cannot proceed")')]
        )
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
                make_response(
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
        assert output[0].parts[0].args == ("from subprocess",)

    def test_registered_function(self):
        """Registered functions survive pickle and work cross-process."""
        llm = Dummy(
            [make_response(thinking="call it", code="task_success(double(21))")]
        )
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

    def test_multi_iteration_cross_process(self):
        """Process isolation handles multi-turn task flow correctly.

        Each emission runs in its own subprocess; the loop drives a
        sequence of independent emissions rather than carrying state
        between them.
        """
        llm = Dummy(
            [
                make_response(thinking="step 1", code="print('looking around')"),
                make_response(thinking="step 2", code="task_success(20)"),
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

    def test_print_event_logged_cross_process(self):
        """Prints from a subprocess emission flow into the event log."""
        from agex.agent.events import OutputEvent
        from agex.eval.objects import PrintAction
        from agex.state import events as state_events

        llm = Dummy(
            [
                make_response(
                    thinking="print and finish",
                    code='print("hello from subprocess")\ntask_success("ok")',
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
        def emit():
            """Emit something."""
            pass

        emit(session="s")
        state = agent._host.resolve_state(config, "s")
        outputs = [e for e in state_events(state) if isinstance(e, OutputEvent)]
        prints = [
            p
            for ev in outputs
            for p in ev.parts
            if isinstance(p, PrintAction) and p.args == ("hello from subprocess",)
        ]
        assert len(prints) == 1

    def test_view_image_cross_process(self):
        """view_image works cross-process via __outputs__."""
        llm = Dummy(
            [
                make_response(
                    thinking="view then succeed",
                    code='view_image("test_img", detail="low")\ntask_continue()',
                ),
                make_response(
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


class TestNestedIsolationValidation:
    """Validate that nested process isolation is rejected at registration time."""

    def setup_method(self):
        clear_agent_registry()

    def test_both_process_raises(self):
        parent = Agent(name="parent", isolation="process")
        child = Agent(name="child", isolation="process")

        @child.task
        def sub_task() -> str:
            """Sub task."""
            pass

        with pytest.raises(ValueError, match="cannot nest"):
            parent.fn()(sub_task)

    def test_kernel_parent_process_child_raises(self):
        parent = Agent(name="parent", isolation="kernel")
        child = Agent(name="child", isolation="process")

        @child.task
        def sub_task() -> str:
            """Sub task."""
            pass

        with pytest.raises(ValueError, match="cannot nest"):
            parent.fn()(sub_task)

    def test_process_parent_none_child_ok(self):
        """Parent isolated, child in-process — allowed."""
        parent = Agent(name="parent", isolation="process")
        child = Agent(name="child", isolation="none")

        @child.task
        def sub_task() -> str:
            """Sub task."""
            pass

        parent.fn()(sub_task)  # Should not raise

    def test_none_parent_kernel_child_ok(self):
        """Parent in-process, child isolated — the recommended pattern."""
        parent = Agent(name="parent", isolation="none")
        child = Agent(name="child", isolation="kernel")

        @child.task
        def sub_task() -> str:
            """Sub task."""
            pass

        parent.fn()(sub_task)  # Should not raise


@_skip_process
class TestHierarchicalIsolation:
    """E2E: orchestrator(none) + sub-agent(process) through dual-decorator."""

    def setup_method(self):
        clear_agent_registry()

    def test_orchestrator_none_specialist_process(self):
        """Recommended pattern: in-process orchestrator delegates to isolated specialist."""
        config = connect_state(type="versioned", storage="memory")

        specialist = Agent(
            name="specialist",
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )
        orchestrator = Agent(
            name="orchestrator",
            state=config,
            isolation="none",
        )

        @orchestrator.fn(docstring="Compute a value")
        @specialist.task("Compute the requested value")
        def compute(expression: str) -> float:
            """Compute a value."""
            pass

        @orchestrator.task("Solve the problem")
        def solve(problem: str) -> dict:
            """Solve problem using specialist."""
            pass

        specialist.llm = Dummy(
            [make_response(thinking="compute", code="task_success(42.0)")]
        )
        orchestrator.llm = Dummy(
            [
                make_response(
                    thinking="delegate",
                    code='result = compute("6 * 7")\ntask_success({"answer": result})',
                )
            ]
        )

        result = solve(problem="What is 6 * 7?", session="s")
        assert result == {"answer": 42.0}

    def test_specialist_with_numpy(self):
        """Specialist uses numpy in process-isolated sandbox."""
        import numpy as np

        config = connect_state(type="versioned", storage="memory")

        specialist = Agent(
            name="np_specialist",
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )
        specialist.module(np, recursive=True, visibility="low")

        orchestrator = Agent(
            name="np_orchestrator",
            state=config,
            isolation="none",
        )

        @orchestrator.fn(docstring="Generate data")
        @specialist.task("Generate numpy data")
        def make_data(prompt: str) -> list:
            """Generate data arrays."""
            pass

        @orchestrator.task("Orchestrate data generation")
        def run(prompt: str) -> list:
            """Run the workflow."""
            pass

        specialist.llm = Dummy(
            [
                make_response(
                    thinking="generate arrays",
                    code="import numpy as np\narr = np.arange(10, dtype=float)\ntask_success([arr])",
                )
            ]
        )
        orchestrator.llm = Dummy(
            [
                make_response(
                    thinking="delegate",
                    code='data = make_data("test")\ntask_success(data)',
                )
            ]
        )

        result = run(prompt="test data", session="s")
        assert len(result) == 1
        assert list(result[0]) == list(range(10))

    def test_orchestrator_none_specialist_kernel(self):
        """Same pattern but with kernel isolation on the specialist."""
        config = connect_state(type="versioned", storage="memory")

        specialist = Agent(
            name="specialist_k",
            state=config,
            isolation="kernel",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )
        orchestrator = Agent(
            name="orchestrator_k",
            state=config,
            isolation="none",
        )

        @orchestrator.fn(docstring="Compute a value")
        @specialist.task("Compute the requested value")
        def compute(expression: str) -> float:
            """Compute a value."""
            pass

        @orchestrator.task("Solve the problem")
        def solve(problem: str) -> dict:
            """Solve problem using specialist."""
            pass

        specialist.llm = Dummy(
            [make_response(thinking="compute", code="task_success(42.0)")]
        )
        orchestrator.llm = Dummy(
            [
                make_response(
                    thinking="delegate",
                    code='result = compute("6 * 7")\ntask_success({"answer": result})',
                )
            ]
        )

        result = solve(problem="What is 6 * 7?", session="s")
        assert result == {"answer": 42.0}


@pytest.mark.asyncio
class TestAsyncSubAgentTask:
    """Async sub-agent task calls are transparently awaited from sandbox code."""

    def setup_method(self):
        clear_agent_registry()

    async def test_async_sub_agent_auto_awaited(self):
        """Orchestrator calls async specialist — coroutine is auto-awaited."""
        config = connect_state(type="versioned", storage="memory")
        specialist = Agent(name="async_spec", state=config, isolation="none")
        orchestrator = Agent(name="async_orch", state=config, isolation="none")

        @orchestrator.fn(docstring="Compute a value")
        @specialist.task("Compute the requested value")
        async def compute(expression: str) -> float:
            """Compute."""
            pass

        @orchestrator.task("Solve the problem")
        async def solve(problem: str) -> dict:
            """Solve."""
            pass

        specialist.llm = Dummy(
            [make_response(thinking="compute", code="task_success(42.0)")]
        )
        orchestrator.llm = Dummy(
            [
                make_response(
                    thinking="delegate",
                    code='result = compute("6 * 7")\ntask_success({"answer": result})',
                )
            ]
        )

        result = await solve(problem="What is 6 * 7?", session="s")
        assert result == {"answer": 42.0}

    async def test_async_hierarchical_multi_specialist(self):
        """Mirrors hierarchical_async.py: orchestrator delegates to two async specialists."""
        config = connect_state(type="versioned", storage="memory")

        data_maker = Agent(name="async_data", state=config, isolation="none")
        plotter = Agent(name="async_plot", state=config, isolation="none")
        orchestrator = Agent(name="async_orch2", state=config, isolation="none")

        @orchestrator.fn(docstring="Generate data arrays")
        @data_maker.task("Generate data")
        async def make_data(prompt: str) -> list:
            """Generate data."""
            pass

        @orchestrator.fn(docstring="Plot the data")
        @plotter.task("Plot data")
        async def plot_data(prompt: str, data: list) -> str:
            """Plot data and return path."""
            pass

        @orchestrator.task("Orchestrate")
        async def run(idea: str) -> str:
            """Orchestrate data generation and plotting."""
            pass

        data_maker.llm = Dummy(
            [make_response(thinking="gen", code="task_success([1, 2, 3])")]
        )
        plotter.llm = Dummy(
            [make_response(thinking="plot", code='task_success("plot.png")')]
        )
        orchestrator.llm = Dummy(
            [
                make_response(
                    thinking="delegate to both",
                    code=(
                        'data = make_data("seasonal")\n'
                        'path = plot_data("umbrella sales", data)\n'
                        "task_success(path)"
                    ),
                )
            ]
        )

        result = await run(idea="umbrella sales", session="s")
        assert result == "plot.png"


class TestCacheUnderProcessIsolation:
    """The agent's ``cache`` must work the same in any isolation mode.

    Under ``isolation="process"``, the worker namespace gets a
    ``RemoteCache`` (sandtrap >= 0.2.1) that proxies operations to
    the parent's live ``Cache(state)`` over an RPC channel.  Reads
    see what the parent has cached; writes propagate back to the
    parent's session cache.
    """

    def setup_method(self):
        clear_agent_registry()

    def test_cache_write_propagates_to_parent_state(self):
        """A subprocess ``cache[k] = v`` lands in the parent's state."""
        from agex.cache import PREFIX

        llm = Dummy(
            [
                make_response(
                    thinking="cache it",
                    code='cache["answer"] = 42\ntask_success("ok")',
                )
            ]
        )
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_cache_write",
            llm=llm,
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def stash() -> str:
            """Stash a value."""
            pass

        assert stash(session="s") == "ok"

        # The write hopped from worker → parent via RPC.  The
        # parent's state has the qualified key.
        state = agent._host.resolve_state(config, "s")
        assert state.get(PREFIX + "answer") == 42

    def test_cache_read_sees_parent_state(self):
        """Pre-cached values are visible to subprocess reads."""
        from agex.cache import PREFIX

        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_cache_read",
            llm=Dummy(),
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        # Pre-populate the cache outside the agent — the agent's
        # subprocess should see it via the RPC proxy.
        state = agent._host.resolve_state(config, "s")
        state[PREFIX + "secret"] = "shibboleth"
        state.commit()

        agent.llm = Dummy(
            [
                make_response(
                    thinking="recall",
                    code='task_success(cache["secret"])',
                )
            ]
        )

        @agent.task
        def recall() -> str:
            """Recall."""
            pass

        assert recall(session="s") == "shibboleth"

    def test_cache_persists_across_isolated_tasks(self):
        """Two task calls in the same session, both isolated.

        Task 1 writes; task 2 reads.  Mirrors the in-process
        ``test_persistence_across_tasks`` semantics under process
        isolation.
        """
        config = connect_state(type="versioned", storage="memory")
        agent = Agent(
            name="proc_cache_persist",
            llm=Dummy(),
            state=config,
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )

        @agent.task
        def stash() -> None:
            """Stash."""
            pass

        @agent.task
        def recall() -> int:
            """Recall."""
            pass

        agent.llm.responses = [
            make_response(
                thinking="store",
                code='cache["answer"] = 42\ntask_success(None)',
            )
        ]
        stash(session="s")

        agent.llm.responses = [
            make_response(thinking="recall", code='task_success(cache["answer"])')
        ]
        assert recall(session="s") == 42

    def test_cache_unpicklable_value_raises_in_subprocess(self):
        """Validation runs on the parent — unpicklable writes raise
        ``CacheError`` and the worker re-raises in the agent's call
        site, just like in-process."""
        # Register a host-side function that produces an unpicklable
        # object (a thread Lock).  Write to cache → handler runs
        # cloudpickle-equivalent picklability check → raises.
        import threading

        from agex.cache import CacheError

        agent = Agent(
            name="proc_cache_bad",
            llm=Dummy(),
            state=connect_state(type="versioned", storage="memory"),
            isolation="process",
            eval_tick_limit=None,
            eval_timeout_seconds=10.0,
        )
        agent.fn(threading.Lock, name="make_lock")
        agent.llm = Dummy(
            [
                make_response(
                    thinking="bad write",
                    code='cache["bad"] = make_lock()',
                )
            ]
        )

        @agent.task
        def attempt() -> None:
            """Try a bad write."""
            pass

        # CacheError surfaces as the task's recoverable error and the
        # task eventually times out (Dummy keeps replaying the same
        # broken code).  Either way, the failure is loud, not silent.
        with pytest.raises((CacheError, TaskTimeout)):
            attempt(session="s")
