"""Test for parent.submodule attribute access in policy resolution."""

import os

from agex.agent.policy.policy import AgentPolicy
from agex.eval.objects import AgexModule
from agex.eval.resolver import Resolver


class MockAgent:
    """Minimal mock agent for resolver tests."""

    def __init__(self, policy):
        self._policy = policy
        self.fingerprint = "test-agent"


def test_submodules_auto_injected_on_registration():
    """Test that registering os.path after os auto-injects submodule reference."""
    policy = AgentPolicy()

    # Register parent first
    policy.register_module(module=os, include=["listdir", "remove"])

    # Register submodule
    policy.register_module(module=os.path, include=["exists", "isfile"])

    # Check that os namespace now has path in submodules
    os_ns = policy.namespaces.get("os")
    assert os_ns is not None
    assert "path" in os_ns.submodules
    # On Unix, os.path is actually posixpath
    assert os_ns.submodules["path"] in ["posixpath", "ntpath"]


def test_submodule_attribute_resolution():
    """Test that resolver returns AgexModule for registered submodule attributes."""
    policy = AgentPolicy()

    # Register both modules
    policy.register_module(module=os, include=["listdir"])
    policy.register_module(module=os.path, include=["exists"])

    # Create resolver and test attribute access
    agent = MockAgent(policy)
    resolver = Resolver(agent)

    # Create AgexModule for 'os'
    os_module = AgexModule(name="os", agent_fingerprint="test-agent")

    # Resolve os.path attribute
    result = resolver.resolve_attribute(os_module, "path", node=None)

    # Should return AgexModule for the registered submodule
    assert isinstance(result, AgexModule)
    # Name will be posixpath on Unix, ntpath on Windows
    assert result.name in ["posixpath", "ntpath"]


def test_submodule_registration_order_independent():
    """Test that registration order doesn't matter."""
    # Register submodule before parent
    policy = AgentPolicy()
    policy.register_module(module=os.path, include=["exists"])
    policy.register_module(module=os, include=["listdir"])

    # Should still inject submodule
    os_ns = policy.namespaces.get("os")
    assert os_ns is not None
    assert "path" in os_ns.submodules


def test_from_import_still_works():
    """Test that from os.path import exists still resolves correctly."""
    policy = AgentPolicy()
    policy.register_module(module=os, include=["listdir"])
    policy.register_module(module=os.path, include=["exists"])

    agent = MockAgent(policy)
    resolver = Resolver(agent)

    # This is how 'from os.path import exists' resolves
    result = resolver.import_from("os.path", "exists", node=None)

    # Should return the exists function
    assert callable(result)
    assert result.__name__ == "exists"


if __name__ == "__main__":
    test_submodules_auto_injected_on_registration()
    print("✓ test_submodules_auto_injected_on_registration passed")

    test_submodule_attribute_resolution()
    print("✓ test_submodule_attribute_resolution passed")

    test_submodule_registration_order_independent()
    print("✓ test_submodule_registration_order_independent passed")

    test_from_import_still_works()
    print("✓ test_from_import_still_works passed")

    print("\nAll tests passed!")
