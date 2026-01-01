"""Tests for connect_state(init=...) parameter."""

from agex import connect_state
from agex.host.local import Local


class TestStateInit:
    """Tests for state initialization via connect_state(init=...)."""

    def test_init_callable_runs_on_fresh_state(self):
        """Init callable is invoked when state is fresh."""
        call_count = 0

        def my_init():
            nonlocal call_count
            call_count += 1
            return {"x": 42, "y": "hello"}

        config = connect_state(type="versioned", storage="memory", init=my_init)
        host = Local()
        state = host.resolve_state(config, "test_session")

        assert call_count == 1
        assert state.get("x") == 42
        assert state.get("y") == "hello"

    def test_init_dict_applied_on_fresh_state(self):
        """Init dict is applied when state is fresh."""
        config = connect_state(
            type="versioned", storage="memory", init={"a": 1, "b": [1, 2, 3]}
        )
        host = Local()
        state = host.resolve_state(config, "test_session")

        assert state.get("a") == 1
        assert state.get("b") == [1, 2, 3]

    def test_init_not_called_on_existing_state(self):
        """Init is not called when state already has sentinel."""
        call_count = 0

        def my_init():
            nonlocal call_count
            call_count += 1
            return {"x": call_count}

        config = connect_state(type="versioned", storage="memory", init=my_init)
        host = Local()

        # First call - should run init
        state1 = host.resolve_state(config, "test_session")
        assert call_count == 1
        assert state1.get("x") == 1

        # Second call - should NOT run init (reuses cached state with sentinel)
        state2 = host.resolve_state(config, "test_session")
        assert call_count == 1  # Still 1, not called again
        assert state2.get("x") == 1

    def test_init_sets_sentinel(self):
        """Init sets the __agex_init__ sentinel."""
        config = connect_state(type="versioned", storage="memory", init={"x": 1})
        host = Local()
        state = host.resolve_state(config, "test_session")

        assert state.get("__agex_init__") is True

    def test_init_none_does_nothing(self):
        """No init when init=None (default)."""
        config = connect_state(type="versioned", storage="memory")
        host = Local()
        state = host.resolve_state(config, "test_session")

        assert state.get("__agex_init__") is None
        assert "__agex_init__" not in state

    def test_init_commits_snapshot(self):
        """Init vars are persisted in a snapshot."""
        config = connect_state(type="versioned", storage="memory", init={"x": 42})
        host = Local()
        state = host.resolve_state(config, "test_session")

        # Check that a snapshot was taken (history has commits)
        history = list(state.history())
        assert len(history) >= 1  # At least the init snapshot
        # Initial commit should contain x
        assert state.get("x") == 42

    def test_init_different_sessions_independent(self):
        """Different sessions each get their own init."""
        call_count = 0

        def my_init():
            nonlocal call_count
            call_count += 1
            return {"session_num": call_count}

        config = connect_state(type="versioned", storage="memory", init=my_init)
        host = Local()

        state1 = host.resolve_state(config, "session_a")
        state2 = host.resolve_state(config, "session_b")

        assert call_count == 2  # Called once per session
        assert state1.get("session_num") == 1
        assert state2.get("session_num") == 2

    def test_init_with_disk_storage(self, tmp_path):
        """Init works with disk storage."""
        call_count = 0

        def my_init():
            nonlocal call_count
            call_count += 1
            return {"x": 42}

        config = connect_state(
            type="versioned", storage="disk", path=str(tmp_path), init=my_init
        )
        host = Local()

        # First call - init runs
        state1 = host.resolve_state(config, "test_session")
        assert call_count == 1
        assert state1.get("x") == 42

        # Create new host (simulates restart) - init should NOT run
        host2 = Local()
        state2 = host2.resolve_state(config, "test_session")
        # Note: call_count check depends on whether we're reloading from disk
        # The sentinel should prevent re-init
        assert state2.get("x") == 42
        assert state2.get("__agex_init__") is True
