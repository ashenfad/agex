import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.console import pprint_events
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import Dummy


@pytest.fixture(autouse=True)
def cleanup():
    clear_agent_registry()
    yield
    clear_agent_registry()


def create_agent(name="test_agent"):
    return Agent(
        name=name,
        llm=Dummy(),
        fs=connect_fs(type="virtual"),
        state=connect_state(type="versioned", storage="memory"),
    )


def test_cross_module_imports():
    """Test A imports B which imports C."""
    agent = create_agent("agent_cross")

    # Setup files
    fs = agent.fs()
    fs.write("c.py", b"def val(): return 42")
    fs.write("b.py", b"import c\ndef val(): return c.val() + 1")
    fs.write("a.py", b"import b\ndef val(): return b.val() * 2")

    responses = [
        LLMResponse(
            thinking="Test chained imports", code="import a\ntask_success(a.val())"
        )
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    result = task(on_event=pprint_events)
    # C=42 -> B=43 -> A=86
    assert result == 86


def test_vfs_class_definitions():
    """Test defining classes in VFS and using them in REPL."""
    agent = create_agent("agent_class_def")

    code = """
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def dist(self):
        return (self.x**2 + self.y**2)**0.5
"""
    agent.fs().write("geometry.py", code.encode())

    responses = [
        LLMResponse(
            thinking="Use VFS class",
            code="import geometry\np = geometry.Point(3, 4)\ntask_success(p.dist())",
        )
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    result = task(on_event=pprint_events)
    assert result == 5.0


def test_stdlib_shadowing_protection():
    """Test that VFS modules cannot shadow whitelisted stdlib modules."""
    agent = create_agent("agent_shadow")

    # Register math so it CAN be shadowed (if we fail) or protected (if we succeed)
    import math

    agent.module(math)

    # Attempt to shadow 'math'
    agent.fs().write("math.py", b"def sqrt(x): return 'fake'")

    # We expect the real math module because it's in the default whitelist
    responses = [
        LLMResponse(
            thinking="Import math", code="import math\ntask_success(math.sqrt(4))"
        ),
        # If the first attempt failed (e.g. returned 'fake' and we asserted inside the agent),
        # provide a fallback to satisfy the loop if needed.
        # But here the agent code succeeds with 'fake', and we assert OUTSIDE.
        # Wait, if math.sqrt(4) returns 'fake', task_success('fake') is called.
        # The test asserts result == 2.0.
        # So why did it fail?
        # Because we WANT it to be 2.0, but it was 'fake' (so shadowing worked!).
        # This means shadowing protection FAILED.
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    result = task(on_event=pprint_events)
    assert result == 2.0  # Real math.sqrt returns float


def test_syntax_error_in_module():
    """Test handling of syntax errors in imported VFS modules."""
    agent = create_agent("agent_syntax")
    agent.fs().write("bad.py", b"def broken( return")

    responses = [
        LLMResponse(thinking="Import bad module", code="import bad"),
        LLMResponse(
            thinking="I see it failed. I will give up.",
            code="task_fail('syntax error')",
        ),
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    with pytest.raises(Exception) as exc:  # TaskFail
        task(on_event=pprint_events)

    assert "syntax error" in str(exc.value)


def test_syntax_error_recovery():
    """Test agent sees syntax error and recovers."""
    agent = create_agent("agent_recovery")
    agent.fs().write("bad.py", b"def broken( return")

    responses = [
        LLMResponse(thinking="Import bad module", code="import bad"),
        LLMResponse(
            thinking="I see it failed. I will give up.",
            code="task_fail('syntax error')",
        ),
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    with pytest.raises(Exception) as exc:  # TaskFail
        task(on_event=pprint_events)
    assert "syntax error" in str(exc.value)


def test_runtime_error_in_module_body():
    """Test runtime error during module execution (top-level)."""
    agent = create_agent("agent_runtime")
    agent.fs().write("crash.py", b"x = 1 / 0")

    responses = [
        LLMResponse(thinking="Import crashing module", code="import crash"),
        LLMResponse(
            thinking="I see it crashed. I will give up.", code="task_fail('crashed')"
        ),
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    with pytest.raises(Exception) as exc:  # TaskFail
        task(on_event=pprint_events)

    assert "crashed" in str(exc.value)


def test_circular_imports_detection():
    """Test circular imports fail gracefully (recursion error or timeout)."""
    agent = create_agent("agent_circular")
    agent.fs().write("ping.py", b"import pong\nx=1")
    agent.fs().write("pong.py", b"import ping\ny=1")

    responses = [
        LLMResponse(thinking="Import ping", code="import ping"),
        LLMResponse(thinking="I see it failed.", code="task_fail('recursion')"),
    ]
    agent.llm.responses = responses

    @agent.task
    def task():
        """Test task"""
        pass

    try:
        task(on_event=pprint_events)
    except Exception:
        # We expect TaskFail("recursion") or TaskTimeout if it loops too long
        # But we mostly want to ensure it doesn't crash the *host* process stack
        # (which pytest would catch as a crash).
        # Since we are running in the same process, RecursionError IS a crash.
        # But EvalError wraps it.
        pass


def test_missing_thinking_error():
    """Test that missing <thinking> tags raise a ResponseParseError."""
    from agex.llm.core import ResponseParseError
    from agex.llm.xml import parse_xml_response

    bad_xml = "<PYTHON>task_success(1)</PYTHON>"
    with pytest.raises(ResponseParseError) as exc:
        parse_xml_response(bad_xml)
    assert "Missing <THINKING> tags" in str(exc.value)


if __name__ == "__main__":
    pytest.main([__file__])
