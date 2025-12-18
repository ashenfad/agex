"""Tests for agex_primer_override functionality."""

from agex import Agent, clear_agent_registry
from agex.agent.primer_text import BUILTIN_PRIMER


def setup_module():
    clear_agent_registry()


def teardown_module():
    clear_agent_registry()


def test_default_primer_usage():
    """Test that agent uses BUILTIN_PRIMER by default."""
    agent = Agent(name="default_agent")

    # We can inspect the method directly
    system_message = agent._build_system_message()

    assert BUILTIN_PRIMER in system_message
    assert "Override" not in system_message


def test_override_primer_usage():
    """Test that agent uses agex_primer_override when provided."""
    custom_primer = "You are a custom agent override."
    agent = Agent(name="override_agent", agex_primer_override=custom_primer)

    system_message = agent._build_system_message()

    assert custom_primer in system_message
    assert BUILTIN_PRIMER not in system_message


def test_override_primer_none_explicit():
    """Test that explicit None falls back to default."""
    agent = Agent(name="none_override_agent", agex_primer_override=None)

    system_message = agent._build_system_message()

    assert BUILTIN_PRIMER in system_message
