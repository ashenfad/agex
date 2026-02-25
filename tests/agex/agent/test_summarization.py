"""Tests for event log summarization."""

import pytest
from kvgit import Namespaced, Staged, Versioned

from agex import Agent, clear_agent_registry
from agex.agent.events import ActionEvent, SummaryEvent, TaskStartEvent
from agex.agent.summarization import (
    SummarizationError,
    maybe_summarize_event_log,
)
from agex.llm.dummy_client import Dummy
from agex.state import _agex_decoder, _agex_encoder
from agex.state.kv import Memory
from agex.state.log import add_event_to_log, get_events_from_log


def _make_state():
    return Staged(Versioned(Memory()), encoder=_agex_encoder, decoder=_agex_decoder)


class TestEventLogSummarization:
    """Test suite for event log summarization."""

    def setup_method(self):
        """Clear agent registry before each test."""
        clear_agent_registry()

    def test_no_summarization_when_disabled(self):
        """Test that summarization doesn't run when not configured."""
        llm = Dummy()
        agent = Agent(name="test", llm=llm)  # No log_high_water_tokens
        state = _make_state()

        # Add many events
        for i in range(10):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        # Should not summarize (no high_water set)
        maybe_summarize_event_log(agent, state, system_message="")

        events = get_events_from_log(state)
        assert len(events) == 10
        assert all(isinstance(e, ActionEvent) for e in events)

    def test_no_summarization_below_high_water(self):
        """Test that summarization doesn't run when below threshold."""
        llm = Dummy()
        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=10000,  # Very high threshold
        )
        state = _make_state()

        # Add a few events (well below threshold)
        for i in range(3):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        total_tokens = sum(e.full_detail_tokens for e in get_events_from_log(state))
        assert total_tokens < 10000

        # Should not summarize
        maybe_summarize_event_log(agent, state, system_message="")

        events = get_events_from_log(state)
        assert len(events) == 3
        assert all(isinstance(e, ActionEvent) for e in events)

    def test_summarization_when_exceeds_high_water(self):
        """Test that summarization runs when exceeding threshold."""
        # Configure client to return a summary
        llm = Dummy()
        llm.summary_response = "Agent performed calculations and stored results."

        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=100,  # Low threshold for testing
            log_low_water_tokens=50,
        )
        state = _make_state()

        # Add events until we exceed threshold
        for i in range(10):
            event = ActionEvent(
                agent_name="test",
                thinking=f"I will compute value {i}",
                code=f"result_{i} = {i} * {i}",
            )
            add_event_to_log(state, event)

        initial_events = get_events_from_log(state)
        initial_count = len(initial_events)
        total_tokens = sum(e.full_detail_tokens for e in initial_events)

        # Verify we're above high water
        assert total_tokens > 100

        # Run summarization
        maybe_summarize_event_log(agent, state, system_message="")

        # Verify summarization occurred
        events = get_events_from_log(state)
        assert len(events) < initial_count  # Fewer events now

        # First event should be a summary
        assert isinstance(events[0], SummaryEvent)
        assert events[0].summarized_event_count > 0
        assert events[0].summary == "Agent performed calculations and stored results."

        # Verify we're now under low water
        new_total_tokens = sum(e.full_detail_tokens for e in events)
        assert new_total_tokens <= 50

    def test_summarization_with_default_low_water(self):
        """Test that low_water defaults to 50% of high_water."""
        llm = Dummy()
        llm.summary_response = "Summary of events."

        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=100,
            # No log_low_water_tokens specified
        )

        # Verify default is 50%
        assert agent.log_low_water_tokens == 50

        state = _make_state()

        # Add events to exceed high water
        for i in range(10):
            event = ActionEvent(
                agent_name="test", thinking=f"thinking {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        maybe_summarize_event_log(agent, state, system_message="")

        # Should be under or near 50 tokens now (may be slightly over due to summary overhead)
        events = get_events_from_log(state)
        total_tokens = sum(e.full_detail_tokens for e in events)
        assert total_tokens <= 70  # Allow some headroom for summary event overhead

    def test_summarization_includes_all_event_types(self):
        """Test that summarization includes TaskStart, setup, and all events."""
        llm = Dummy()
        llm.summary_response = "Complete summary."

        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=100,
        )
        state = _make_state()

        # Add TaskStartEvent
        task_start = TaskStartEvent(
            agent_name="test",
            task_name="test_task",
            inputs={},
            message="Compute the answer",
        )
        add_event_to_log(state, task_start)

        # Add setup event
        setup_action = ActionEvent(
            agent_name="test", thinking="Setup", code="import math", source="setup"
        )
        add_event_to_log(state, setup_action)

        # Add regular events
        for i in range(8):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        initial_events = get_events_from_log(state)
        assert any(isinstance(e, TaskStartEvent) for e in initial_events)
        assert any(
            e.source == "setup" for e in initial_events if isinstance(e, ActionEvent)
        )

        maybe_summarize_event_log(agent, state, system_message="")

        # All old events including TaskStart and setup should be summarized
        events = get_events_from_log(state)
        assert isinstance(events[0], SummaryEvent)
        assert (
            events[0].summarized_event_count >= 3
        )  # At least TaskStart + setup + some actions

    def test_summarization_error_propagation(self):
        """Test that LLM failures raise SummarizationError."""
        # Configure client to fail
        llm = Dummy()
        llm.summary_exception = RuntimeError("LLM service unavailable")

        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=50,
        )
        state = _make_state()

        # Add events to trigger summarization
        for i in range(5):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        # Should raise SummarizationError
        with pytest.raises(SummarizationError, match="Failed to summarize .* events"):
            maybe_summarize_event_log(agent, state, system_message="")

    def test_cascading_summarization(self):
        """Test that summaries can themselves be summarized."""
        llm = Dummy()
        llm.summary_response = "Meta-summary of summaries and events."

        agent = Agent(
            name="test",
            llm=llm,
            log_high_water_tokens=100,
            log_low_water_tokens=50,
        )
        state = _make_state()

        # Add events and trigger first summarization
        for i in range(10):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        maybe_summarize_event_log(agent, state, system_message="")

        events_after_first = get_events_from_log(state)
        assert isinstance(events_after_first[0], SummaryEvent)

        # Add more events to trigger second summarization
        for i in range(10, 20):
            event = ActionEvent(
                agent_name="test", thinking=f"thought {i}", code=f"x = {i}"
            )
            add_event_to_log(state, event)

        # Set new summary response
        llm.summary_response = "Second level summary."
        maybe_summarize_event_log(agent, state, system_message="")

        # Should have new summary that potentially includes the old summary
        events_final = get_events_from_log(state)
        assert isinstance(events_final[0], SummaryEvent)
        # The new summary might have summarized the old summary + some new events

    def test_summarization_validation_errors(self):
        """Test Agent initialization validation for summarization params."""
        llm = Dummy()

        # Test: low without high should raise
        with pytest.raises(
            ValueError, match="log_low_water_tokens requires log_high_water_tokens"
        ):
            Agent(
                name="test1",
                llm=llm,
                log_low_water_tokens=50,  # Missing high_water
            )

        # Test: low >= high should raise
        with pytest.raises(ValueError, match="log_low_water_tokens .* must be <"):
            Agent(
                name="test2",
                llm=llm,
                log_high_water_tokens=100,
                log_low_water_tokens=100,  # Equal to high
            )

        with pytest.raises(ValueError, match="log_low_water_tokens .* must be <"):
            Agent(
                name="test3",
                llm=llm,
                log_high_water_tokens=100,
                log_low_water_tokens=150,  # Greater than high
            )

    def test_summary_event_has_full_namespace_and_commit_hash(self):
        """Test that SummaryEvent gets full_namespace and commit_hash set correctly."""
        llm = Dummy()
        llm.summary_response = "Events were summarized."

        agent = Agent(
            name="test_agent",
            llm=llm,
            log_high_water_tokens=100,
            log_low_water_tokens=50,
        )
        state = _make_state()

        # Snapshot to create a commit
        state.commit()

        # Add events to exceed threshold
        for i in range(10):
            event = ActionEvent(
                agent_name="test_agent",
                thinking=f"thinking {i}",
                code=f"x = {i}",
            )
            add_event_to_log(state, event)

        # Snapshot to commit events
        state.commit()
        commit_before = state.current_commit

        # Run summarization
        maybe_summarize_event_log(agent, state, system_message="")

        # Get the summary event
        events = get_events_from_log(state)
        summary = events[0]
        assert isinstance(summary, SummaryEvent)

        # Verify full_namespace is set to agent name (not empty string)
        assert summary.full_namespace == "test_agent"
        assert summary.full_namespace != ""

        # Verify commit_hash is set
        assert summary.commit_hash is not None
        assert summary.commit_hash == commit_before

    def test_summary_event_has_namespaced_full_namespace(self):
        """Test that SummaryEvent in namespaced state gets correct full_namespace."""
        llm = Dummy()
        llm.summary_response = "Sub-agent events summarized."

        agent = Agent(
            name="sub_agent",
            llm=llm,
            log_high_water_tokens=100,
            log_low_water_tokens=50,
        )

        # Create nested namespaced state (simulating hierarchical agents)
        state = _make_state()
        parent_state = Namespaced(state, "parent")
        ns_state = Namespaced(parent_state, "sub_agent")

        # Add events to exceed threshold
        for i in range(10):
            event = ActionEvent(
                agent_name="sub_agent",
                thinking=f"thinking {i}",
                code=f"x = {i}",
            )
            add_event_to_log(ns_state, event)

        # Verify events have the namespaced full_namespace
        initial_events = get_events_from_log(ns_state)
        assert all(e.full_namespace == "parent/sub_agent" for e in initial_events)

        # Run summarization
        maybe_summarize_event_log(agent, ns_state, system_message="")

        # Get the summary event
        events = get_events_from_log(ns_state)
        summary = events[0]
        assert isinstance(summary, SummaryEvent)

        # Verify full_namespace is set to the full namespace path (not just agent name)
        assert summary.full_namespace == "parent/sub_agent"
        assert summary.agent_name == "sub_agent"
