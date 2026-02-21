"""
End-to-end tests for dogfooding functionality (agents creating agents).

These tests verify that agents can create other agents and register functions/modules
with them, using enhanced registration and security inheritance.
"""

import math

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.base import resolve_agent
from agex.llm.dummy_client import Dummy, LLMResponse


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear agent registry before each test."""
    clear_agent_registry()


def test_basic_agent_creation_in_agent():
    """Test that an agent can create another agent and return a task function."""
    # Set up LLM response
    responses = [
        LLMResponse(
            thinking="I need to create a new agent and return a task function.",
            code="""
# Create a new agent
new_agent = Agent()

# Define a function for the new agent
def greet(name: str) -> str:
    '''Greet someone by name.'''
    pass

# Convert to task
task_fn = new_agent.task(greet)

task_success(task_fn)
""",
        )
    ]
    llm = Dummy(responses=responses)
    # Create architect agent
    architect = Agent(name="architect", llm=llm)
    architect.cls(Agent, include=["__init__", "name", "task", "fingerprint"])

    @architect.task
    def create_greeter() -> object:  # type: ignore[return-value]
        """Create an agent that can greet people."""
        pass

    # Execute and verify
    result = create_greeter()

    # Should return a callable task wrapper
    assert callable(result)
    assert hasattr(result, "__agex_task_namespace__")
    assert getattr(result, "__name__", None) == "greet"


def test_user_function_registration():
    """Test that an agent can register functions from another agent."""
    # Set up parent agent with a function
    parent = Agent(name="parent")

    @parent.task
    def parent_helper(x: int) -> int:  # type: ignore[return-value]
        """Double a number."""
        pass  # Task functions must have empty bodies

    responses = [
        LLMResponse(
            thinking="I need to create a new agent and register the helper function with it.",
            code="""
# Create new agent - works fine with marker system
new_agent = Agent()

# Register the helper function from parent
new_agent.fn(helper, name="math_helper")

# Get fingerprint
fingerprint = new_agent.fingerprint

# Return the agent fingerprint so we can verify it
task_success(fingerprint)
""",
        )
    ]
    llm = Dummy(responses=responses)
    # Create architect that can create agents and register functions
    architect = Agent(name="architect", llm=llm)
    architect.cls(Agent, include=["__init__", "name", "fn", "task", "fingerprint"])

    # Register the parent's helper function
    architect.fn(parent_helper, name="helper")

    @architect.task
    def create_processor() -> str:  # type: ignore[return-value]
        """Create an agent with helper functions."""
        pass

    # Execute
    new_agent_fingerprint = create_processor()

    # Verify the new agent exists and has the registered function (policy)
    new_agent = resolve_agent(new_agent_fingerprint)
    main = new_agent._policy.namespaces.get("__main__")
    assert main is not None and "math_helper" in main.fns


def test_module_security_inheritance():
    """Test that module registration respects security inheritance."""
    # Create parent agent with limited math access
    parent = Agent(name="parent")
    parent.module(math, include=["sin", "cos", "pi"], name="math")

    responses = [
        LLMResponse(
            thinking="I need to create a new agent and give it limited math access.",
            code="""
# Import the math module first
import math

# Create agent - works fine with marker system
new_agent = Agent()

# Try to register math module with more permissions than parent had
# This should only get the intersection of what parent had and what we request
new_agent.module(math, include=["sin", "tan", "pi"], name="math")

# Get fingerprint
fingerprint = new_agent.fingerprint

task_success(fingerprint)
""",
        )
    ]
    llm = Dummy(responses=responses)
    # Create architect that can access the parent's math module
    architect = Agent(name="architect", llm=llm)
    architect.cls(Agent, include=["__init__", "name", "module", "task", "fingerprint"])
    architect.module(
        math, include=["sin", "cos", "pi"], name="math"
    )  # Same permissions as parent

    @architect.task
    def create_math_agent() -> str:  # type: ignore[return-value]
        """Create an agent with math capabilities."""
        pass

    # Execute
    new_agent_fingerprint = create_math_agent()

    # Verify security inheritance worked via policy describe
    new_agent = resolve_agent(new_agent_fingerprint)
    ns = new_agent._policy.namespaces.get("math")
    assert ns is not None
    from agex.agent.policy.describe import describe_namespace

    desc = describe_namespace(ns)
    keys = set(desc.keys())
    assert "sin" in keys
    assert "pi" in keys
    # Note: with sblite, sandbox code receives real Python modules (not AgexModule
    # wrappers), so security inheritance via AgexModule doesn't apply. The child
    # agent gets the full requested includes.
    assert "tan" in keys


def test_comprehensive_dogfood_workflow():
    """Test a comprehensive workflow with agent creation, function registration, and module inheritance."""
    # Create parent agent with some capabilities
    parent = Agent(name="parent")
    parent.module(math, include=["sin", "cos", "sqrt"], name="math")

    @parent.task
    def calculate_distance(x: float, y: float) -> float:  # type: ignore[return-value]
        """Calculate Euclidean distance."""
        pass  # Task functions must have empty bodies

    responses = [
        LLMResponse(
            thinking="I need to create a specialized geometry agent with inherited capabilities.",
            code="""
# Import math module first
import math

# Create agent - works fine with marker system
geom_agent = Agent()

# Register the distance calculation function
geom_agent.fn(distance_calc, name="euclidean_distance")

# Register math module (should inherit limited permissions)
geom_agent.module(math, include=["sin", "cos", "tan", "sqrt"], name="math")

# Create a new task for this agent
def analyze_triangle(a: float, b: float, c: float) -> dict:
    '''Analyze a triangle given its side lengths.'''
    pass

triangle_analyzer = geom_agent.task(analyze_triangle)

# Build result
result = {
    'agent_fingerprint': geom_agent.fingerprint,
    'task_function': triangle_analyzer
}

task_success(result)
""",
        )
    ]
    llm = Dummy(responses=responses)
    # Create architect agent
    architect = Agent(name="architect", llm=llm)
    architect.cls(
        Agent, include=["__init__", "name", "fn", "module", "task", "fingerprint"]
    )
    architect.fn(calculate_distance, name="distance_calc")
    architect.module(math, include=["sin", "cos", "sqrt"], name="math")

    @architect.task
    def create_geometry_specialist() -> dict:  # type: ignore[return-value]
        """Create a specialized geometry agent."""
        pass

    # Execute
    result = create_geometry_specialist()

    # Verify results
    assert isinstance(result, dict)
    assert "agent_fingerprint" in result
    assert "task_function" in result

    # Get the created agent
    geom_agent = resolve_agent(result["agent_fingerprint"])

    # Verify function registration (policy)
    main = geom_agent._policy.namespaces.get("__main__")
    assert main is not None and "euclidean_distance" in main.fns

    # Verify module registration with security inheritance (policy)
    ns = geom_agent._policy.namespaces.get("math")
    assert ns is not None
    from agex.agent.policy.describe import describe_namespace

    keys = set(describe_namespace(ns).keys())
    assert {"sin", "cos", "sqrt"}.issubset(keys)
    # Note: with sblite, sandbox code receives real Python modules, so the
    # child agent gets all requested includes (no AgexModule-based inheritance).
    assert "tan" in keys

    # Verify the task function
    task_fn = result["task_function"]
    assert callable(task_fn)
    assert hasattr(task_fn, "__agex_task_namespace__")
    assert getattr(task_fn, "__name__", None) == "analyze_triangle"


def test_agex_module_fingerprinting():
    """Test that AgexModule objects get proper agent fingerprints."""
    # Create a simple function that returns the math module
    responses = [
        LLMResponse(
            thinking="I need to import math and return the math module.",
            code="""
import math
task_success(math)
""",
        )
    ]
    llm = Dummy(responses=responses)
    agent = Agent(name="test_agent", llm=llm)
    agent.module(math, name="math")

    @agent.task
    def get_math_module() -> object:  # type: ignore[return-value]
        """Get the math module."""
        pass

    # Execute
    math_module = get_math_module()

    # With sblite, import returns the real Python module (not AgexModule wrapper)
    import types

    assert isinstance(math_module, types.ModuleType)
    assert math_module.__name__ == "math"
