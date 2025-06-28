import math

from agex import Agent

# 1. Create an agent with a clear role and iteration limit
mathy_agent = Agent(
    primer="You are an expert at using a calculator. You are given a math problem and your goal is to return just the numeric answer.",
    max_iterations=5,
)

# 2. Register the math module with medium visibility
# (shows function signatures but not docstrings to save context)
mathy_agent.module(math, visibility="medium")


# 3. Define a task that the agent will solve
@mathy_agent.task
def run_calculation(problem: str) -> float:  # type: ignore
    """
    Solve the user's math problem and return a number.
    """
    pass


# 4. Run the agent with a specific problem
if __name__ == "__main__":
    result = run_calculation(
        "What is the square root of 256, multiplied by pi?",
    )
    print(result)
