"""
End-to-end tests for dogfooding functionality (agents creating agents).

These tests verify that agents can create other agents and register functions/modules
with them, using the TaskUserFunction, enhanced registration, and security inheritance.
"""

import math

import pytest

from agex import Agent, clear_agent_registry
from agex.agent.base import resolve_agent
from agex.eval.functions import TaskUserFunction
from agex.eval.objects import AgexModule
from agex.llm.dummy_client import DummyLLMClient, LLMResponse


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear agent registry before each test."""
    clear_agent_registry()


def test_basic_agent_creation_in_agent():
    """Test that an agent can create another agent and return a TaskUserFunction."""
    # Set up LLM response
    responses = [
        LLMResponse(
            thinking="I need to create a new agent and return a task function.",
            code="""
# Create a new agent
with Agent() as new_agent:
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
    llm_client = DummyLLMClient(responses=responses)
    # Create architect agent
    architect = Agent(name="architect", llm_client=llm_client)
    architect.cls(Agent, include=["__init__", "name", "task", "fingerprint"])

    @architect.task
    def create_greeter() -> object:  # type: ignore[return-value]
        """Create an agent that can greet people."""
        pass

    # Execute and verify
    result = create_greeter()

    # Should return a TaskUserFunction
    assert isinstance(result, TaskUserFunction)
    assert result.name == "greet"
    assert result.task_agent_fingerprint != architect.fingerprint
    assert result.agent_fingerprint != result.task_agent_fingerprint


def test_user_function_registration():
    """Test that an agent can register UserFunctions from another agent."""
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
# Use context manager to avoid pickle issues
with Agent() as new_agent:
    # Register the helper function from parent
    new_agent.fn(helper, name="math_helper")
    
    # Extract fingerprint before leaving context
    fingerprint = new_agent.fingerprint

# Return the agent fingerprint so we can verify it
task_success(fingerprint)
""",
        )
    ]
    llm_client = DummyLLMClient(responses=responses)
    # Create architect that can create agents and register functions
    architect = Agent(name="architect", llm_client=llm_client)
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

# Use context manager to avoid pickle issues
with Agent() as new_agent:
    # Try to register math module with more permissions than parent had
    # This should only get the intersection of what parent had and what we request
    new_agent.module(math, include=["sin", "tan", "pi"], name="math")
    
    # Extract fingerprint before leaving context
    fingerprint = new_agent.fingerprint

task_success(fingerprint)
""",
        )
    ]
    llm_client = DummyLLMClient(responses=responses)
    # Create architect that can access the parent's math module
    architect = Agent(name="architect", llm_client=llm_client)
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
    assert "tan" not in keys


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

# Use context manager to avoid pickle issues
with Agent() as geom_agent:
    # Register the distance calculation function
    geom_agent.fn(distance_calc, name="euclidean_distance")
    
    # Register math module (should inherit limited permissions)
    geom_agent.module(math, include=["sin", "cos", "tan", "sqrt"], name="math")
    
    # Create a new task for this agent
    def analyze_triangle(a: float, b: float, c: float) -> dict:
        '''Analyze a triangle given its side lengths.'''
        pass
    
    triangle_analyzer = geom_agent.task(analyze_triangle)
    
    # Extract data before leaving context
    result = {
        'agent_fingerprint': geom_agent.fingerprint,
        'task_function': triangle_analyzer
    }

    task_success(result)
""",
        )
    ]
    llm_client = DummyLLMClient(responses=responses)
    # Create architect agent
    architect = Agent(name="architect", llm_client=llm_client)
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
    assert "tan" not in keys

    # Verify the task function
    task_fn = result["task_function"]
    assert isinstance(task_fn, TaskUserFunction)
    assert task_fn.name == "analyze_triangle"
    assert task_fn.task_agent_fingerprint == geom_agent.fingerprint


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
    llm_client = DummyLLMClient(responses=responses)
    agent = Agent(name="test_agent", llm_client=llm_client)
    agent.module(math, name="math")

    @agent.task
    def get_math_module() -> object:  # type: ignore[return-value]
        """Get the math module."""
        pass

    # Execute
    math_module = get_math_module()

    # Should be an AgexModule with proper fingerprint
    assert isinstance(math_module, AgexModule)
    assert math_module.name == "math"
    assert math_module.agent_fingerprint == agent.fingerprint
