from agex.agent.events import ActionEvent, OutputEvent, SuccessEvent, TaskStartEvent
from agex.state import Namespaced, Versioned, events, kv
from agex.state.log import add_event_to_log


class TestUnifiedEventsAPI:
    """Test the unified events() API that consolidates event access."""

    def test_events_current_namespace_with_children_default(self):
        """Test events() on current namespace includes children by default."""
        state = Versioned(kv.Memory())

        # Create namespace hierarchy
        ns_root = Namespaced(state, "root")
        ns_worker = Namespaced(ns_root, "worker")

        # Add events to root
        add_event_to_log(
            ns_root,
            TaskStartEvent(
                agent_name="root", task_name="orchestrate", inputs={}, message="start"
            ),
        )
        add_event_to_log(
            ns_root, ActionEvent(agent_name="root", thinking="planning", code="plan()")
        )

        # Add events to worker sub-namespace
        add_event_to_log(
            ns_worker,
            TaskStartEvent(
                agent_name="worker", task_name="work", inputs={}, message="work"
            ),
        )
        add_event_to_log(ns_worker, SuccessEvent(agent_name="worker", result="done"))

        # Test: events() from root should include children by default
        all_events = events(ns_root)
        assert len(all_events) == 4  # 2 root + 2 worker events

        # Verify we got events from both namespaces
        agent_names = {e.agent_name for e in all_events if hasattr(e, "agent_name")}
        assert agent_names == {"root", "worker"}

    def test_events_current_namespace_children_false(self):
        """Test events() with children=False only shows current namespace."""
        state = Versioned(kv.Memory())

        # Create namespace hierarchy
        ns_root = Namespaced(state, "root")
        ns_worker = Namespaced(ns_root, "worker")

        # Add events to root
        add_event_to_log(
            ns_root,
            TaskStartEvent(
                agent_name="root", task_name="orchestrate", inputs={}, message="start"
            ),
        )
        add_event_to_log(
            ns_root, ActionEvent(agent_name="root", thinking="planning", code="plan()")
        )

        # Add events to worker sub-namespace
        add_event_to_log(
            ns_worker,
            TaskStartEvent(
                agent_name="worker", task_name="work", inputs={}, message="work"
            ),
        )
        add_event_to_log(ns_worker, SuccessEvent(agent_name="worker", result="done"))

        # Test: events() with children=False should only show root events
        root_only_events = events(ns_root, children=False)
        assert len(root_only_events) == 2  # Only root events

        # Verify we only got root events
        agent_names = {
            e.agent_name for e in root_only_events if hasattr(e, "agent_name")
        }
        assert agent_names == {"root"}

    def test_events_namespace_navigation(self):
        """Test events() can navigate to specific namespaces."""
        state = Versioned(kv.Memory())

        # Create complex hierarchy: root -> orchestrator -> worker
        ns_root = Namespaced(state, "root")
        ns_orch = Namespaced(ns_root, "orchestrator")
        ns_worker = Namespaced(ns_orch, "worker")

        # Add events to each level
        add_event_to_log(
            ns_root,
            TaskStartEvent(
                agent_name="root", task_name="start", inputs={}, message="root"
            ),
        )
        add_event_to_log(
            ns_orch,
            ActionEvent(
                agent_name="orchestrator", thinking="planning", code="orchestrate()"
            ),
        )
        add_event_to_log(
            ns_worker, SuccessEvent(agent_name="worker", result="completed")
        )

        # Test: Navigate directly to orchestrator namespace
        orch_events_result = events(state, "root", "orchestrator")
        assert (
            len(orch_events_result) == 2
        )  # orchestrator + worker (children=True default)

        # Test: Navigate directly to worker namespace
        worker_events_result = events(state, "root", "orchestrator", "worker")
        assert len(worker_events_result) == 1  # Only worker events

        # Test: Navigate to orchestrator without children
        orch_only_events = events(state, "root", "orchestrator", children=False)
        assert len(orch_only_events) == 1  # Only orchestrator events

    def test_events_versioned_and_live_states(self):
        """Test events() works correctly with Versioned and Live states."""
        from agex.state import Live

        # Test with Versioned state
        versioned_state = Versioned(kv.Memory())
        add_event_to_log(
            versioned_state,
            TaskStartEvent(
                agent_name="versioned", task_name="task", inputs={}, message="test"
            ),
        )

        result = events(versioned_state)
        assert len(result) == 1
        assert result[0].agent_name == "versioned"

        # Test with Live state
        live_state = Live()
        add_event_to_log(
            live_state,
            ActionEvent(agent_name="live", thinking="thinking", code="code()"),
        )

        result = events(live_state)
        assert len(result) == 1
        assert result[0].agent_name == "live"

    def test_events_hierarchical_collection(self):
        """Test events() correctly collects from complex namespace hierarchies."""
        state = Versioned(kv.Memory())

        # Create multi-level hierarchy with multiple branches
        ns_app = Namespaced(state, "app")
        ns_workers = Namespaced(ns_app, "workers")
        ns_worker1 = Namespaced(ns_workers, "worker1")
        ns_worker2 = Namespaced(ns_workers, "worker2")
        ns_cache = Namespaced(ns_app, "cache")

        # Add events at different levels
        add_event_to_log(
            ns_app,
            TaskStartEvent(
                agent_name="app", task_name="start", inputs={}, message="app"
            ),
        )
        add_event_to_log(
            ns_workers,
            ActionEvent(agent_name="workers", thinking="managing", code="manage()"),
        )
        add_event_to_log(
            ns_worker1, SuccessEvent(agent_name="worker1", result="result1")
        )
        add_event_to_log(
            ns_worker2, SuccessEvent(agent_name="worker2", result="result2")
        )
        add_event_to_log(ns_cache, OutputEvent(agent_name="cache", parts=["cached"]))

        # Test: Get all events from app (should include all descendants)
        all_app_events = events(state, "app")
        assert len(all_app_events) == 5  # All events from entire hierarchy

        # Test: Get events from workers namespace (should include worker1 and worker2)
        workers_events_result = events(state, "app", "workers")
        assert len(workers_events_result) == 3  # workers + worker1 + worker2

        # Test: Get events from workers without children
        workers_only = events(state, "app", "workers", children=False)
        assert len(workers_only) == 1  # Only workers events

    def test_events_empty_namespaces(self):
        """Test events() handles empty namespaces correctly."""
        state = Versioned(kv.Memory())

        # Test empty root state
        empty_events = events(state)
        assert empty_events == []

        # Test empty namespaced state
        ns_empty = Namespaced(state, "empty")
        empty_ns_events = events(ns_empty)
        assert empty_ns_events == []

        # Test navigation to non-existent namespace
        nonexistent_events = events(state, "nonexistent", "path")
        assert nonexistent_events == []

    def test_events_key_filtering(self):
        """Test that events() only processes __event_log__ keys."""
        state = Versioned(kv.Memory())
        ns_test = Namespaced(state, "test")

        # Add event log and other keys
        add_event_to_log(
            ns_test,
            TaskStartEvent(
                agent_name="test", task_name="task", inputs={}, message="test"
            ),
        )
        ns_test.set("other_data", "not_events")
        ns_test.set("config", {"setting": "value"})

        # Add nested namespace with mixed keys
        ns_worker = Namespaced(ns_test, "worker")
        add_event_to_log(ns_worker, SuccessEvent(agent_name="worker", result="done"))
        state.set("test/worker/task_data", "not_events")
        state.set("test/cache/item", "also_not_events")

        # Test: Should only collect from __event_log__ keys
        all_events = events(ns_test)
        assert (
            len(all_events) == 2
        )  # test event + worker event, ignoring non-event keys

        # Verify event content
        agent_names = {e.agent_name for e in all_events if hasattr(e, "agent_name")}
        assert agent_names == {"test", "worker"}


def test_events_chronological_sorting():
    """Test that events() returns events sorted by timestamp in chronological order."""
    from datetime import datetime, timezone

    state = Versioned(kv.Memory())

    # Create events with known timestamps (simulate events created at different times)
    base_time = datetime.now(timezone.utc)

    # Create events with explicit timestamps out of order
    event1 = TaskStartEvent(
        agent_name="agent1", task_name="task1", inputs={}, message="first"
    )
    event2 = ActionEvent(agent_name="agent2", thinking="second", code="code2()")
    event3 = SuccessEvent(agent_name="agent3", result="third")
    event4 = OutputEvent(agent_name="agent4", parts=["fourth"])

    # Manually set timestamps in reverse chronological order to test sorting
    event1.timestamp = base_time.replace(microsecond=100000)  # Earliest
    event2.timestamp = base_time.replace(microsecond=200000)  # Second
    event3.timestamp = base_time.replace(microsecond=300000)  # Third
    event4.timestamp = base_time.replace(microsecond=400000)  # Latest

    # Store events in different namespaces in random order
    ns1 = Namespaced(state, "ns1")
    ns2 = Namespaced(state, "ns2")

    # Add events out of chronological order
    add_event_to_log(ns1, event3)  # Third
    add_event_to_log(ns1, event1)  # First
    add_event_to_log(ns2, event4)  # Fourth
    add_event_to_log(ns2, event2)  # Second

    # Get all events - should be sorted chronologically regardless of storage order
    all_events = events(state)

    # Verify we got all 4 events
    assert len(all_events) == 4

    # Verify chronological ordering by timestamp
    timestamps = [e.timestamp for e in all_events]
    assert timestamps == sorted(timestamps), "Events should be sorted chronologically"

    # Verify the specific order is correct
    expected_agent_order = ["agent1", "agent2", "agent3", "agent4"]
    actual_agent_order = [e.agent_name for e in all_events]
    assert (
        actual_agent_order == expected_agent_order
    ), f"Expected {expected_agent_order}, got {actual_agent_order}"

    # Test with children=False still maintains sorting
    ns1_events = events(state, "ns1", children=False)
    assert len(ns1_events) == 2
    ns1_timestamps = [e.timestamp for e in ns1_events]
    assert ns1_timestamps == sorted(
        ns1_timestamps
    ), "Namespace events should be sorted chronologically"
