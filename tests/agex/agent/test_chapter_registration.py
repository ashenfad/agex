"""Tests for chapter task registration."""

import pytest

from agex import CHAPTER_TASK, Agent, clear_agent_registry
from agex.agent.chapter import Chapter


@pytest.fixture(autouse=True)
def clear_registry():
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestChapterTaskRegistration:
    def test_registered_when_water_marks_set(self):
        agent = Agent(name="ch_test1", log_high_water_tokens=100000)
        assert agent._chapter_task is not None

    def test_not_registered_without_water_marks(self):
        agent = Agent(name="ch_test2")
        assert agent._chapter_task is None

    def test_chapter_task_name(self):
        agent = Agent(name="ch_test3", log_high_water_tokens=100000)
        assert agent._chapter_task._task_name == CHAPTER_TASK

    def test_chapter_class_constructable(self):
        """Chapter class should be registered as constructable."""
        agent = Agent(name="ch_test4", log_high_water_tokens=100000)
        # Check that Chapter is in the policy's registered classes
        found = False
        for ns in agent._policy.namespaces.values():
            if "Chapter" in ns.classes:
                entry = ns.classes["Chapter"]
                assert entry.cls is Chapter
                assert entry.constructable is True
                found = True
                break
        assert found, "Chapter class not found in agent's registered classes"
