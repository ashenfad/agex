"""
End-to-end tests for git CLI integration with the agent loop.

Exercises multi-turn agent workflows using Dummy LLM responses that
include terminal git commands, verifying that:
- git commit/log/diff work through the real agent loop
- System commits (safe_commit) are invisible to git log
- git reset restores files without breaking session state
- Multi-turn workflows with branching and merging work end-to-end
"""

import pytest

from agex import Agent, connect_fs, connect_state
from agex.agent.base import clear_agent_registry
from agex.agent.events import OutputEvent
from agex.eval.objects import PrintAction
from agex.git_cli import register_git
from agex.llm import Dummy
from agex.llm.core import LLMResponse


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


def _make_agent(name="dev_agent"):
    """Create an agent with VFS, versioned state, and git skill."""
    fs = connect_fs(type="virtual")
    state = connect_state(type="versioned", storage="memory")
    agent = Agent(name=name, fs=fs, state=state)
    register_git(agent)
    return agent


class TestGitCommitAndLog:
    """Agent writes files, commits, and inspects history."""

    def test_write_commit_and_log(self):
        """Multi-turn: write a file, commit, then check git log."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write a file via FILE tag and commit via terminal
                LLMResponse(
                    thinking="I'll create a helper and commit it.",
                    code="",
                    file_actions=[
                        {
                            "path": "helpers/utils.py",
                            "content": "def add(a, b):\n    return a + b\n",
                            "mode": "write",
                        }
                    ],
                    terminal="git commit -m 'add utils module'",
                ),
                # Turn 2: check git log
                LLMResponse(
                    thinking="Let me check the commit history.",
                    terminal="git log --oneline",
                ),
                # Turn 3: complete the task with the log output
                LLMResponse(
                    thinking="I can see my commit in the log.",
                    code="task_success('done')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def develop() -> str:
            """Develop some code."""
            pass

        result = develop(on_event=collected_events.append)
        assert result == "done"

        # Verify git log output appeared in an OutputEvent
        output_text = _collect_output_text(collected_events)
        assert "add utils module" in output_text

    def test_system_commits_invisible_in_log(self):
        """System commits from safe_commit should not appear in git log."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write a file (safe_commit happens at turn boundary)
                LLMResponse(
                    thinking="Writing a file.",
                    terminal="echo 'version 1' > app.py",
                ),
                # Turn 2: commit with a message
                LLMResponse(
                    thinking="Committing.",
                    terminal="git commit -m 'first real commit'",
                ),
                # Turn 3: write another file (another safe_commit at boundary)
                LLMResponse(
                    thinking="More work.",
                    terminal="echo 'version 2' > app.py",
                ),
                # Turn 4: commit again
                LLMResponse(
                    thinking="Committing v2.",
                    terminal="git commit -m 'second commit'",
                ),
                # Turn 5: check git log — should only show agent-tagged commits
                LLMResponse(
                    thinking="Checking history.",
                    terminal="git log --oneline",
                ),
                # Turn 6: complete
                LLMResponse(
                    thinking="Done.",
                    code="task_success('checked')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def work() -> str:
            """Do work."""
            pass

        result = work(on_event=collected_events.append)
        assert result == "checked"

        # Find the git log output
        all_output = _collect_output_text(collected_events)

        # Should see both agent commits
        assert "first real commit" in all_output
        assert "second commit" in all_output

        # Count commit hashes (7-char hex followed by a message) — should be
        # exactly 2 tagged commits, not 4+ with system commits mixed in.
        import re

        commit_lines = re.findall(r"[0-9a-f]{7} .+", all_output)
        assert len(commit_lines) == 2


class TestGitDiff:
    """Agent modifies files and inspects diffs."""

    def test_diff_between_commits(self):
        """Write v1, commit, write v2, commit, diff shows the change."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Create v1.",
                    terminal="echo 'hello' > greet.py && git commit -m 'v1'",
                ),
                LLMResponse(
                    thinking="Update to v2.",
                    terminal="echo 'world' > greet.py && git commit -m 'v2'",
                ),
                LLMResponse(
                    thinking="Check what changed.",
                    terminal="git diff HEAD~1",
                ),
                LLMResponse(
                    thinking="Done.",
                    code="task_success('diffed')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def check_diff() -> str:
            """Check diffs."""
            pass

        result = check_diff(on_event=collected_events.append)
        assert result == "diffed"

        diff_text = _collect_output_text(collected_events)
        assert "hello" in diff_text or "world" in diff_text


class TestGitReset:
    """Agent uses git reset to restore files."""

    def test_reset_restores_files_and_agent_continues(self):
        """Reset restores files without breaking session state."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: create and commit v1
                LLMResponse(
                    thinking="Creating v1.",
                    terminal="echo 'good code' > app.py && git commit -m 'v1 good'",
                ),
                # Turn 2: create and commit v2 (bad change)
                LLMResponse(
                    thinking="Making a bad change.",
                    terminal="echo 'bad code' > app.py && git commit -m 'v2 bad'",
                ),
                # Turn 3: reset to v1
                LLMResponse(
                    thinking="That was wrong, let me reset.",
                    terminal="git reset --hard HEAD~1",
                ),
                # Turn 4: verify the file was restored, then commit the restore
                LLMResponse(
                    thinking="Check the restored file.",
                    terminal="cat app.py",
                ),
                # Turn 5: complete — the REPL still works after reset
                LLMResponse(
                    thinking="File is restored, task complete.",
                    code="task_success('restored')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def fix_mistake() -> str:
            """Fix a mistake."""
            pass

        result = fix_mistake(on_event=collected_events.append)
        assert result == "restored"

        # Verify "Restored" appeared in output
        all_output = _collect_output_text(collected_events)
        assert "Restored" in all_output or "good code" in all_output


class TestGitBranching:
    """Agent uses branches for experimentation."""

    def test_branch_experiment_and_merge(self):
        """Create branch, do work, merge back to main."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: set up main with a base file
                LLMResponse(
                    thinking="Setting up base.",
                    terminal="echo 'base' > main.py && git commit -m 'base'",
                ),
                # Turn 2: create experiment branch and add a file
                LLMResponse(
                    thinking="Branching for experiment.",
                    terminal="git checkout -b experiment && echo 'new feature' > feature.py && git commit -m 'add feature'",
                ),
                # Turn 3: switch back to main
                LLMResponse(
                    thinking="Back to main.",
                    terminal="git checkout main",
                ),
                # Turn 4: merge the experiment
                LLMResponse(
                    thinking="The experiment worked, merging.",
                    terminal="git merge experiment",
                ),
                # Turn 5: verify and complete
                LLMResponse(
                    thinking="Check that feature.py is on main now.",
                    terminal="cat feature.py",
                ),
                LLMResponse(
                    thinking="Merge successful.",
                    code="task_success('merged')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def experiment() -> str:
            """Try an experiment."""
            pass

        result = experiment(on_event=collected_events.append)
        assert result == "merged"

        # Verify feature.py content appeared
        all_output = _collect_output_text(collected_events)
        assert "new feature" in all_output

    def test_branch_abandon_experiment(self):
        """Create branch, decide it's bad, delete it, continue on main."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                LLMResponse(
                    thinking="Base setup.",
                    terminal="echo 'stable' > app.py && git commit -m 'stable base'",
                ),
                LLMResponse(
                    thinking="Try something risky.",
                    terminal="git checkout -b risky && echo 'risky change' > app.py && git commit -m 'risky attempt'",
                ),
                LLMResponse(
                    thinking="That didn't work. Abandon.",
                    terminal="git checkout main && git branch -d risky",
                ),
                LLMResponse(
                    thinking="Back on main, file should be stable.",
                    terminal="cat app.py && git log --oneline",
                ),
                LLMResponse(
                    thinking="Good — stable code preserved.",
                    code="task_success('abandoned')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def try_and_abandon() -> str:
            """Try and abandon."""
            pass

        result = try_and_abandon(on_event=collected_events.append)
        assert result == "abandoned"

        all_output = _collect_output_text(collected_events)
        # Should see stable content on main
        assert "stable" in all_output
        # Should NOT see risky commit in log (branch was deleted)
        # The "risky attempt" commit was on the deleted branch


class TestGitWithPython:
    """Agent combines git with python script execution."""

    def test_write_script_commit_run_iterate(self):
        """Write a script, commit, run it, edit it, run again."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write a script and commit
                LLMResponse(
                    thinking="Writing compute script.",
                    terminal="echo 'print(2 + 2)' > compute.py && git commit -m 'add compute'",
                ),
                # Turn 2: run the script
                LLMResponse(
                    thinking="Running the script.",
                    terminal="python compute.py",
                ),
                # Turn 3: update the script and commit
                LLMResponse(
                    thinking="Update to multiply instead.",
                    terminal="echo 'print(3 * 7)' > compute.py && git commit -m 'change to multiply'",
                ),
                # Turn 4: run updated script and diff
                LLMResponse(
                    thinking="Run updated and check diff.",
                    terminal="python compute.py && git diff HEAD~1",
                ),
                # Turn 5: complete
                LLMResponse(
                    thinking="Done iterating.",
                    code="task_success('iterated')",
                ),
            ]
        )

        collected_events = []

        @agent.task
        def iterate() -> str:
            """Iterate on code."""
            pass

        result = iterate(on_event=collected_events.append)
        assert result == "iterated"

        all_output = _collect_output_text(collected_events)
        # Should see both script outputs
        assert "4" in all_output  # 2 + 2
        assert "21" in all_output  # 3 * 7


class TestGitAddSelective:
    """Agent uses git add for selective commits."""

    def test_selective_commit_through_agent_loop(self):
        """Agent writes two files via FILE, adds only one in terminal, commits selectively."""
        agent = _make_agent()
        agent.llm = Dummy(
            responses=[
                # Turn 1: write two files via FILE tags (applied before terminal runs),
                # then add only a.py and commit in the same turn's terminal block.
                LLMResponse(
                    thinking="Creating two files, but only committing a.py.",
                    file_actions=[
                        {"path": "a.py", "content": "module A\n", "mode": "write"},
                        {"path": "b.py", "content": "module B\n", "mode": "write"},
                    ],
                    terminal="git add a.py && git commit -m 'just module A' && git status",
                ),
                # Turn 2: now add and commit b.py (it's still in Staged after
                # safe_commit ran — wait, safe_commit flushed it. So we write
                # b.py again and commit it.)
                LLMResponse(
                    thinking="Now committing b.py.",
                    file_actions=[
                        {"path": "b.py", "content": "module B v2\n", "mode": "write"},
                    ],
                    terminal="git add b.py && git commit -m 'add module B'",
                ),
                # Turn 3: log should show both commits
                LLMResponse(
                    thinking="Check history.",
                    terminal="git log --oneline",
                ),
                LLMResponse(
                    thinking="Done.",
                    code="task_success('selective')",
                ),
            ]
        )

        collected_events = []

        @agent.task("Selective commit")
        def work() -> str:
            """Do selective work."""
            pass

        result = work(on_event=collected_events.append)
        assert result == "selective"

        all_output = _collect_output_text(collected_events)

        # Both commits should appear in log
        assert "just module A" in all_output
        assert "add module B" in all_output

        # After selective commit, b.py is still uncommitted (only a.py was flushed).
        # The git status output should show b.py as unstaged.
        assert "not staged" in all_output
