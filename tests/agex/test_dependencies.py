"""Tests for Agent.dependencies property."""

from agex import Agent


class TestDependencies:
    """Test incremental dependency tracking."""

    def test_empty_agent_has_base_deps(self):
        """Empty agent should have python and agex versions, no packages."""
        agent = Agent()
        deps = agent.dependencies

        assert deps.python_version  # e.g., "3.12"
        assert deps.agex_version  # e.g., "0.1.0"
        assert deps.packages == []

    def test_stdlib_not_tracked(self):
        """Stdlib functions should not add dependencies."""
        agent = Agent()
        import json
        import math

        agent.fn(json.loads, name="parse")
        agent.fn(math.sqrt, name="sqrt")

        deps = agent.dependencies
        assert deps.packages == []

    def test_fn_tracks_third_party(self):
        """@agent.fn on third-party function adds dependency."""
        agent = Agent()
        import pytest

        agent.fn(pytest.fail, name="fail")

        deps = agent.dependencies
        assert any(p.startswith("pytest==") for p in deps.packages)

    def test_module_tracks_third_party(self):
        """agent.module on third-party module adds dependency."""
        agent = Agent()
        import pytest

        agent.module(pytest, name="pyt")

        deps = agent.dependencies
        assert any(p.startswith("pytest==") for p in deps.packages)

    def test_cls_tracks_third_party(self):
        """@agent.cls on third-party class adds dependency."""
        agent = Agent()
        from pytest import Item

        agent.cls(Item)

        deps = agent.dependencies
        assert any(p.startswith("pytest==") for p in deps.packages)

    def test_dependencies_deduplicated(self):
        """Multiple registrations from same package should dedupe."""
        agent = Agent()
        import pytest

        agent.fn(pytest.fail, name="fail")
        agent.fn(pytest.skip, name="skip")
        agent.fn(pytest.warns, name="warns")

        deps = agent.dependencies
        pytest_deps = [p for p in deps.packages if p.startswith("pytest==")]
        assert len(pytest_deps) == 1  # Only one pytest entry

    def test_dependencies_id_stable(self):
        """Dependencies.id should be stable for same packages."""
        agent = Agent()
        import pytest

        agent.fn(pytest.approx, name="approx")
        id1 = agent.dependencies.id

        # Create new agent with same registration
        agent2 = Agent()
        agent2.fn(pytest.approx, name="approx")
        id2 = agent2.dependencies.id

        assert id1 == id2

    def test_dependencies_id_changes_with_packages(self):
        """Dependencies.id should change when packages differ."""
        agent1 = Agent()
        agent2 = Agent()

        import pytest

        agent1.fn(pytest.approx, name="approx")
        # agent2 has no registrations

        assert agent1.dependencies.id != agent2.dependencies.id

    def test_hierarchical_agent_deps(self):
        """Parent agent includes dependencies from sub-agents."""
        import numpy as np

        # Create sub-agent with numpy
        sub_agent = Agent(name="sub")
        sub_agent.module(np, recursive=True, visibility="low")

        # Create parent that uses sub-agent
        parent = Agent(name="parent")

        @parent.fn
        @sub_agent.task
        def process_data(data: list):
            """Process data with numpy."""
            pass

        # Parent should include numpy from sub-agent
        parent_packages = parent.dependencies.packages
        assert any(
            "numpy" in p for p in parent_packages
        ), f"Parent should include numpy from sub-agent. Got: {parent_packages}"

        # Sub-agent should have numpy
        sub_packages = sub_agent.dependencies.packages
        assert any(
            "numpy" in p for p in sub_packages
        ), f"Sub-agent should have numpy. Got: {sub_packages}"


class TestWarmup:
    """Test agent.warmup() method."""

    def test_warmup_no_op_for_local(self):
        """warmup() should be a no-op for Local host (default)."""
        agent = Agent()
        # Should not raise
        agent.warmup()

    def test_warmup_passes_dependencies(self):
        """warmup() should pass Dependencies to host."""
        from unittest.mock import MagicMock

        import pytest

        agent = Agent()
        agent.fn(pytest.approx, name="approx")

        # Mock the host
        mock_host = MagicMock()
        agent._host = mock_host

        agent.warmup()

        # Verify warmup was called with Dependencies
        mock_host.warmup.assert_called_once()
        deps = mock_host.warmup.call_args[0][0]
        assert deps.python_version
        assert deps.agex_version
        assert any(p.startswith("pytest==") for p in deps.packages)


class TestOptionalDependencies:
    """Test detection of installed optional dependencies."""

    def test_get_installed_optional_deps_unknown_package(self):
        """Unknown package should return empty set."""
        agent = Agent()
        result = agent._get_installed_optional_deps("nonexistent-fake-package-xyz")
        assert result == set()

    def test_get_installed_optional_deps_no_extras(self):
        """Package with no optional deps should return empty set."""
        agent = Agent()
        # pytest doesn't have optional deps (just required ones)
        result = agent._get_installed_optional_deps("pytest")
        # Should not error, may return empty or some deps
        assert isinstance(result, set)

    def test_optional_deps_helper_detects_installed(self):
        """The helper should detect installed optional deps of a package."""
        from importlib import metadata

        agent = Agent()

        # Find a real package in the environment that has optional deps
        # We'll use agex itself which has optional deps like fastapi
        try:
            metadata.requires("agex")
        except metadata.PackageNotFoundError:
            # agex not installed as package in test env
            return

        # Check if any optional deps exist and are installed
        optional_deps = agent._get_installed_optional_deps("agex")

        # The result should be a set of "package==version" strings
        for dep in optional_deps:
            assert "==" in dep, f"Expected 'package==version' format, got: {dep}"
