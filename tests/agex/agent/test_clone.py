"""Tests for Agent.clone_registrations() and AgentPolicy.copy()."""

import math

import pytest

from agex import (
    Agent,
    clear_agent_registry,
    connect_fs,
    connect_state,
    run_file_in_sandbox,
)


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear agent registry before and after each test."""
    clear_agent_registry()
    yield
    clear_agent_registry()


class TestAgentCloneRegistrations:
    """Tests for Agent.clone_registrations() class method."""

    def test_clone_registrations_creates_new_agent(self):
        """clone_registrations creates a distinct agent instance."""
        source = Agent(name="source")
        clone = Agent.clone_registrations(source, name="clone")

        assert clone is not source
        assert clone.name == "clone"

    def test_clone_registrations_uses_default_params(self):
        """clone_registrations uses default Agent params, not source values."""
        source = Agent(
            name="source",
            primer="Source primer",
            eval_timeout_seconds=99.0,
            max_iterations=50,
        )
        clone = Agent.clone_registrations(source, name="clone")

        # Should use defaults, not source values
        assert clone.primer is None
        assert clone.eval_timeout_seconds == 5.0
        assert clone.max_iterations == 10

    def test_clone_registrations_accepts_custom_params(self):
        """clone_registrations accepts custom params for the new agent."""
        source = Agent(name="source")
        clone = Agent.clone_registrations(
            source,
            name="clone",
            primer="Custom primer",
            eval_timeout_seconds=15.0,
            max_iterations=25,
        )

        assert clone.primer == "Custom primer"
        assert clone.eval_timeout_seconds == 15.0
        assert clone.max_iterations == 25

    def test_clone_registrations_has_independent_state(self):
        """clone_registrations creates agent with independent state config."""
        state_config = connect_state(type="versioned", storage="memory")
        source = Agent(state=state_config)
        clone = Agent.clone_registrations(source, name="clone")

        # Clone should have default (None/ephemeral), not source's state
        assert clone._state_config is None
        assert clone._state_config is not source._state_config

    def test_clone_registrations_accepts_custom_state(self):
        """clone_registrations can specify its own state config."""
        source = Agent()
        clone_state = connect_state(type="versioned", storage="memory")
        clone = Agent.clone_registrations(source, name="clone", state=clone_state)

        assert clone._state_config is clone_state

    def test_clone_registrations_has_independent_fs(self):
        """clone_registrations creates agent with independent fs config."""
        fs_config = connect_fs(type="virtual")
        source = Agent(fs=fs_config)
        clone = Agent.clone_registrations(source, name="clone")

        # Clone should have default VFS, not source's fs reference
        assert clone._fs_config is not source._fs_config

    def test_clone_registrations_has_independent_host(self):
        """clone_registrations creates agent with independent host."""
        source = Agent()
        clone = Agent.clone_registrations(source, name="clone")

        # Each should have their own host instance
        assert clone._host is not source._host

    def test_clone_registrations_policy_is_independent(self):
        """Modifications to clone's policy don't affect source."""
        source = Agent(name="source")
        source.module(math)

        clone = Agent.clone_registrations(source, name="clone")

        # Verify both have math
        assert "math" in source._policy.namespaces
        assert "math" in clone._policy.namespaces

        # Add json to clone only
        import json

        clone.module(json)

        # Source should NOT have json
        assert "json" not in source._policy.namespaces
        assert "json" in clone._policy.namespaces

    def test_clone_registrations_policy_shares_live_objects(self):
        """Clone's policy shares references to modules/functions."""
        source = Agent(name="source")
        source.module(math)

        clone = Agent.clone_registrations(source, name="clone")

        # The actual module object should be the same reference
        source_ns = source._policy.namespaces["math"]
        clone_ns = clone._policy.namespaces["math"]

        assert source_ns.module is clone_ns.module
        assert source_ns.module is math

    def test_clone_registrations_inherits_registered_functions(self):
        """Clone inherits functions registered on source."""

        def my_func(x: int) -> int:
            return x * 2

        source = Agent(name="source")
        source.fn(my_func)

        clone = Agent.clone_registrations(source, name="clone")

        # Check function is in clone's __main__ namespace
        main_ns = clone._policy.namespaces.get("__main__")
        assert main_ns is not None
        assert "my_func" in main_ns.fn_objects

    def test_clone_registrations_inherits_registered_classes(self):
        """Clone inherits classes registered on source."""

        class MyClass:
            def method(self) -> str:
                return "hello"

        source = Agent(name="source")
        source.cls(MyClass)

        clone = Agent.clone_registrations(source, name="clone")

        # Check class is in clone's __main__ namespace
        main_ns = clone._policy.namespaces.get("__main__")
        assert main_ns is not None
        assert "MyClass" in main_ns.classes

    def test_clone_registrations_tracked_modules_independent(self):
        """Clone has independent tracked modules set."""
        source = Agent(name="source")
        source.module(math)

        clone = Agent.clone_registrations(source, name="clone")

        # Both should have math tracked
        assert "math" in source._tracked_modules
        assert "math" in clone._tracked_modules

        # Add to clone, verify source unchanged
        import json

        clone.module(json)

        assert "json" not in source._tracked_modules
        assert "json" in clone._tracked_modules

    def test_clone_registrations_host_object_registry_copied(self):
        """Clone gets a copy of host object registry."""
        source = Agent(name="source")

        # Manually add something to registry
        source._host_object_registry["test"] = "value"

        clone = Agent.clone_registrations(source, name="clone")

        # Clone should have it
        assert "test" in clone._host_object_registry

        # But modifications should be independent
        clone._host_object_registry["new"] = "other"
        assert "new" not in source._host_object_registry


class TestPolicyCopy:
    """Tests for AgentPolicy.copy() method."""

    def test_copy_creates_new_policy(self):
        """Copy creates a distinct policy instance."""
        source = Agent(name="source")
        source.module(math)

        copy = source._policy.copy()

        assert copy is not source._policy

    def test_copy_namespaces_independent(self):
        """Copied policy has independent namespaces dict."""
        source = Agent(name="source")
        source.module(math)

        copy = source._policy.copy()

        # Add to copy
        import json

        from agex.agent.policy.datatypes import Namespace

        copy.namespaces["json"] = Namespace(
            name="json", kind="module", module=json, visibility="high"
        )

        # Source should not have it
        assert "json" not in source._policy.namespaces

    def test_copy_namespace_objects_independent(self):
        """Individual Namespace objects are copied, not shared."""
        source = Agent(name="source")
        source.module(math)

        copy = source._policy.copy()

        source_ns = source._policy.namespaces["math"]
        copy_ns = copy.namespaces["math"]

        assert source_ns is not copy_ns

    def test_copy_preserves_namespace_attributes(self):
        """Copied namespaces preserve all attributes."""
        source = Agent(name="source")
        source.module(math, visibility="low", include=["sqrt", "floor"])

        copy = source._policy.copy()

        copy_ns = copy.namespaces["math"]
        assert copy_ns.name == "math"
        assert copy_ns.kind == "module"
        assert copy_ns.visibility == "low"
        assert copy_ns.include == ["sqrt", "floor"]
        assert copy_ns.module is math

    def test_copy_configure_dict_independent(self):
        """Configure dict in namespace is copied, not shared."""
        from agex import MemberSpec

        source = Agent(name="source")
        source.module(math, configure={"sqrt": MemberSpec(visibility="high")})

        copy = source._policy.copy()

        source_ns = source._policy.namespaces["math"]
        copy_ns = copy.namespaces["math"]

        # Dicts should be independent
        assert source_ns.configure is not copy_ns.configure

    def test_copy_class_namespaces_preserved(self):
        """Class namespaces mapping is properly copied."""

        class MyClass:
            pass

        source = Agent(name="source")
        source.cls(MyClass)

        copy = source._policy.copy()

        assert MyClass in copy._class_namespaces


class TestRunFileInSandbox:
    """Tests for run_file_in_sandbox() helper function."""

    def test_run_simple_code(self):
        """Run simple code from VFS."""
        agent = Agent(
            name="test",
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
        )
        agent.module(math)

        # Write file
        fs = agent.fs("session")
        fs.write("test.py", b"import math\nresult = math.sqrt(25)")

        # Run
        state = run_file_in_sandbox(agent, "test.py", "session")

        assert state.get("result") == 5.0

    def test_run_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        agent = Agent(
            name="test",
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
        )

        with pytest.raises(FileNotFoundError, match="nonexistent.py"):
            run_file_in_sandbox(agent, "nonexistent.py", "session")

    def test_run_with_custom_timeout(self):
        """Custom timeout is passed through."""
        agent = Agent(
            name="test",
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
            eval_timeout_seconds=1.0,
        )
        agent.module(math)

        fs = agent.fs("session")
        fs.write("test.py", b"import math\nx = 1")

        # Should not raise with reasonable code
        state = run_file_in_sandbox(
            agent, "test.py", "session", eval_timeout_seconds=10.0
        )
        assert state.get("x") == 1

    def test_clone_registrations_for_sandbox(self):
        """Typical use case: clone registrations to create isolated sandbox."""
        # Main agent
        main_agent = Agent(name="main")
        main_agent.module(math)

        # Clone registrations to create sandbox with its own state/fs
        import json

        sandbox = Agent.clone_registrations(
            main_agent,
            name="sandbox",
            state=connect_state(type="versioned", storage="memory"),
            fs=connect_fs(type="virtual"),
        )
        sandbox.module(json)

        # Write app code to sandbox's VFS
        fs = sandbox.fs("session")
        fs.write(
            "app/main.py",
            b"""import math
import json
# Simulate an app that uses both
data = {"sqrt_2": math.sqrt(2)}
result = json.dumps(data)
""",
        )

        # Run in sandbox
        state = run_file_in_sandbox(sandbox, "app/main.py", "session")

        assert state.get("result") == '{"sqrt_2": 1.4142135623730951}'
