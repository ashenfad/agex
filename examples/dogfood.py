"""
Dogfooding Example: Agents Creating Agents

This example demonstrates agex's ability to "eat its own dogfood" by having
agents create other agents at runtime.
"""

from agex import Agent
from agex.llm.dummy_client import DummyLLMClient, LLMResponse


def main():
    print("=== Dogfooding Demo: Agents Creating Agents ===\n")

    # Create an architect agent that can create other agents
    architect = Agent(name="architect")

    # Register the Agent class so the architect can use it
    architect.cls(Agent, include=["name", "primer", "fn", "module", "task"])

    # Set up dummy LLM responses for predictable demo
    architect_responses = [
        LLMResponse(
            thinking="I need to create a new agent and make a task function from it.",
            code="""
# Create a specialist math agent with unique name
with Agent() as math_agent:  # Let it auto-generate a unique name
    
    # Define a function that the new agent will implement
    def solve_equation(equation: str) -> str:
        '''Solve a mathematical equation step by step.'''
        pass
    
    # Make this function a task for the new agent  
    task_function = math_agent.task(solve_equation)

# Return the task function
exit_success(task_function)
""",
        )
    ]

    architect.llm_client = DummyLLMClient(responses=architect_responses)

    @architect.task
    def create_math_specialist() -> object:  # type: ignore[return-value]
        """Create a specialist agent for solving mathematical equations."""
        pass

    print("Step 1: Architect creates a math specialist agent...")
    try:
        math_solver = create_math_specialist()
        print(f"✓ Created math solver: {type(math_solver).__name__}")

        # Check if it's a TaskUserFunction as expected
        from agex.eval.functions import TaskUserFunction

        if isinstance(math_solver, TaskUserFunction):
            print("✓ Successfully created TaskUserFunction!")
            print(f"  - Task name: {math_solver.name}")
            print(f"  - Task agent fingerprint: {math_solver.task_agent_fingerprint}")
            print(f"  - Original agent fingerprint: {math_solver.agent_fingerprint}")
        else:
            print(f"⚠️  Expected TaskUserFunction, got {type(math_solver)}")

    except Exception as e:
        print(f"❌ Error creating math specialist: {e}")

    print("\n=== Demo Complete ===")
    print("This demonstrates the foundational architecture for agents creating agents!")


if __name__ == "__main__":
    main()
