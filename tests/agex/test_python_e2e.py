"""
End-to-end tests for python script execution through the agent loop.

Exercises multi-turn agent workflows using Dummy LLM responses that
combine <FILE> writes, <TERMINAL> python execution, and <PYTHON> REPL
blocks, verifying that:
- Scripts run with the agent's registered modules
- Fresh namespace per script run (no REPL leakage)
- The REPL bridge pattern (develop in scripts, complete from REPL)
- Pipeline composition (python script | grep)
- Script iteration (write, run, edit, run again)
- Combined python + git workflows
"""

import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.events import OutputEvent
from agex.eval.objects import PrintAction
from agex.git_cli import register_git
from agex.helpers.stdlib import register_stdlib
from agex.llm import Dummy
from tests.agex._emissions import make_response


def _collect_output_text(events):
    """Extract all text from OutputEvent PrintAction parts."""
    parts = []
    for e in events:
        if isinstance(e, OutputEvent):
            for part in e.parts:
                if isinstance(part, PrintAction):
                    parts.extend(str(x) for x in part)
                else:
                    parts.append(str(part))
    return "\n".join(parts)


@pytest.fixture(autouse=True)
def clean_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


def _make_agent(name="script_agent", with_git=False):
    """Create an agent with VFS, versioned state, and stdlib."""
    fs = connect_fs(type="virtual")
    state = connect_state(type="versioned", storage="memory")
    agent = Agent(name=name, fs=fs, state=state)
    register_stdlib(agent)
    if with_git:
        register_git(agent)
    return agent


class TestBasicScriptExecution:
    """Agent writes and runs scripts through the loop."""

    def test_write_and_run_script(self):
        """Write a script via FILE, run it via TERMINAL, complete from REPL."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write a script
                make_response(
                    thinking="I'll write a computation script.",
                    file_actions=[
                        {
                            "path": "compute.py",
                            "content": "import math\nresult = math.sqrt(144)\nprint(f'sqrt(144) = {result}')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python compute.py",
                ),
                # Turn 2: complete with the result from REPL
                make_response(
                    thinking="The script printed the answer. Let me return it.",
                    code="task_success(12.0)",
                ),
            ]
        )

        collected = []

        @agent.task("Compute something")
        def compute() -> float:
            """Compute a value."""
            pass

        result = compute(on_event=collected.append)
        assert result == 12.0

        output = _collect_output_text(collected)
        assert "sqrt(144) = 12.0" in output

    def test_inline_python_c(self):
        """Agent uses python -c for a quick check."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                make_response(
                    thinking="Quick math check.",
                    terminal="python -c 'print(7 * 6)'",
                ),
                make_response(
                    thinking="Got it.",
                    code="task_success(42)",
                ),
            ]
        )

        collected = []

        @agent.task("Quick check")
        def check() -> int:
            """Check."""
            pass

        result = check(on_event=collected.append)
        assert result == 42
        assert "42" in _collect_output_text(collected)


class TestScriptIteration:
    """Agent edits and re-runs scripts across turns."""

    def test_write_run_edit_rerun(self):
        """Write v1, run, see output, edit to v2, run again."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write and run v1
                make_response(
                    thinking="Writing first version.",
                    file_actions=[
                        {
                            "path": "analyze.py",
                            "content": "data = [1, 2, 3, 4, 5]\nprint(f'sum = {sum(data)}')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python analyze.py",
                ),
                # Turn 2: edit and run v2
                make_response(
                    thinking="Need the mean instead.",
                    file_actions=[
                        {
                            "path": "analyze.py",
                            "content": "data = [1, 2, 3, 4, 5]\nprint(f'mean = {sum(data)/len(data)}')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python analyze.py",
                ),
                # Turn 3: complete
                make_response(
                    thinking="Both results computed.",
                    code="task_success('iterated')",
                ),
            ]
        )

        collected = []

        @agent.task("Analyze data")
        def analyze() -> str:
            """Analyze."""
            pass

        result = analyze(on_event=collected.append)
        assert result == "iterated"

        output = _collect_output_text(collected)
        assert "sum = 15" in output
        assert "mean = 3.0" in output


class TestREPLBridge:
    """Agent develops in scripts and completes via REPL import."""

    def test_develop_in_script_complete_from_repl(self):
        """The canonical REPL bridge pattern from the primer."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write a module
                make_response(
                    thinking="Writing the solver module.",
                    file_actions=[
                        {
                            "path": "helpers/solver.py",
                            "content": "def solve(x):\n    return x ** 2 + 1\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python -c 'from helpers.solver import solve; print(solve(5))'",
                ),
                # Turn 2: import from REPL and complete
                make_response(
                    thinking="It works. Returning the result via REPL.",
                    code="from helpers.solver import solve\ntask_success(solve(5))",
                ),
            ]
        )

        collected = []

        @agent.task("Solve something")
        def run_solver() -> int:
            """Solve."""
            pass

        result = run_solver(on_event=collected.append)
        assert result == 26  # 5^2 + 1

        output = _collect_output_text(collected)
        assert "26" in output


class TestNamespaceIsolation:
    """Script namespace doesn't leak to/from REPL."""

    def test_script_vars_dont_leak_to_repl(self):
        """Variables defined in a script are not visible in the REPL."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: run a script that defines a variable
                make_response(
                    thinking="Running script with a variable.",
                    file_actions=[
                        {
                            "path": "setup.py",
                            "content": "secret_value = 42\nprint(f'set secret_value = {secret_value}')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python setup.py",
                ),
                # Turn 2: try to access from REPL — should fail
                make_response(
                    thinking="Checking if variable leaked.",
                    code="try:\n    print(secret_value)\n    task_success('leaked!')\nexcept NameError:\n    task_success('isolated')",
                ),
            ]
        )

        collected = []

        @agent.task("Test isolation")
        def test_iso() -> str:
            """Test."""
            pass

        result = test_iso(on_event=collected.append)
        assert result == "isolated"

    def test_repl_vars_dont_leak_to_script(self):
        """Variables defined in the REPL are not visible in scripts."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: define a variable in REPL
                make_response(
                    thinking="Setting a REPL variable.",
                    code="repl_var = 'from_repl'\ntask_continue()",
                ),
                # Turn 2: try to access from script
                make_response(
                    thinking="Checking if REPL var is visible in script.",
                    file_actions=[
                        {
                            "path": "check.py",
                            "content": "try:\n    print(repl_var)\nexcept NameError:\n    print('not visible')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python check.py",
                ),
                # Turn 3: complete
                make_response(
                    thinking="Confirmed isolation.",
                    code="task_success('isolated')",
                ),
            ]
        )

        collected = []

        @agent.task("Test reverse isolation")
        def test_rev() -> str:
            """Test."""
            pass

        result = test_rev(on_event=collected.append)
        assert result == "isolated"

        output = _collect_output_text(collected)
        assert "not visible" in output


class TestPipelineComposition:
    """Scripts compose with termish built-in commands."""

    def test_python_piped_to_grep(self):
        """Script output piped through grep."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                make_response(
                    thinking="Generate lines and filter.",
                    file_actions=[
                        {
                            "path": "gen.py",
                            "content": "for i in range(10):\n    label = 'even' if i % 2 == 0 else 'odd'\n    print(f'{i} is {label}')\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python gen.py | grep even",
                ),
                make_response(
                    thinking="Got the even numbers.",
                    code="task_success('filtered')",
                ),
            ]
        )

        collected = []

        @agent.task("Filter numbers")
        def filter_nums() -> str:
            """Filter."""
            pass

        result = filter_nums(on_event=collected.append)
        assert result == "filtered"

        output = _collect_output_text(collected)
        assert "0 is even" in output
        assert "1 is odd" not in output


class TestPythonWithGit:
    """Combined python + git workflows."""

    def test_script_development_with_git_checkpoints(self):
        """Write script, commit, iterate, diff to see changes."""
        agent = _make_agent(with_git=True)
        agent.llm = Dummy(
            responses=[
                # Turn 1: write v1 and commit
                make_response(
                    thinking="Writing v1 of the processor.",
                    file_actions=[
                        {
                            "path": "process.py",
                            "content": "data = [1, 2, 3]\nprint(sum(data))\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python process.py && git commit -m 'v1: sum'",
                ),
                # Turn 2: update to v2 and commit
                make_response(
                    thinking="Changing to product.",
                    file_actions=[
                        {
                            "path": "process.py",
                            "content": "import math\ndata = [1, 2, 3]\nprint(math.prod(data))\n",
                            "mode": "write",
                        }
                    ],
                    terminal="python process.py && git commit -m 'v2: product'",
                ),
                # Turn 3: check git diff and complete
                make_response(
                    thinking="Let me see what changed.",
                    terminal="git diff HEAD~1 && git log --oneline",
                ),
                make_response(
                    thinking="Done.",
                    code="task_success('developed')",
                ),
            ]
        )

        collected = []

        @agent.task("Develop a processor")
        def develop() -> str:
            """Develop."""
            pass

        result = develop(on_event=collected.append)
        assert result == "developed"

        output = _collect_output_text(collected)
        # Both script outputs should appear
        assert "6" in output  # sum([1,2,3])
        # Git log should show both commits
        assert "v1: sum" in output
        assert "v2: product" in output
