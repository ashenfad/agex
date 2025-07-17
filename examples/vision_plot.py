import io
import os
import pandas as pd
import matplotlib.pyplot as plt

from agex import Agent
from agex.llm.core import LLMResponse

# Note: This example requires pandas and matplotlib to be installed.
# You can install them with:
# pip install "agex[examples,vision]"


# 1. Define the canned responses for our dummy LLM.
plot_code = """
import io
import pandas as pd
import matplotlib.pyplot as plt

# Use the registered modules to work with the data. The input data is available in the 'inputs' object.
df = pd.read_csv(io.StringIO(inputs.data))

# Create the plot
plt.figure(figsize=(8, 5))
plt.bar(df['Month'], df['Sales'])
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()

# Use view_image() to "see" the plot. This makes the image available to the LLM for the next turn.
view_image(plt.gcf())
"""
analysis_code = 'task_success("The bar chart shows monthly sales, with a peak in June and the lowest point in January.")'

responses = [
    LLMResponse(
        thinking="I will plot the data using pandas and matplotlib.", code=plot_code
    ),
    LLMResponse(
        thinking="I have seen the plot. Now I will describe the trend.",
        code=analysis_code,
    ),
]

# 2. Create the main agent for our application, configuring it to use the dummy provider.
main_agent = Agent(
    primer="""
You are a data analyst. Your goal is to plot the data you are given, view the plot, and then describe the trend you see.
You have access to the `pd` (pandas) and `plt` (matplotlib.pyplot) modules.
You must call `view_image()` on the plot you create.
Finally, you must call `task_success()` with a string describing the plot.
""",
    # Pass the provider and responses directly to the agent constructor.
    llm_provider="dummy",
    responses=responses,
)

# 3. Register the modules the agent is allowed to use.
main_agent.module(pd)
main_agent.module(plt)
main_agent.module(io)


# 4. Define a task for the agent.
@main_agent.task
def analyze_data(data: str):
    """
    Analyzes a CSV string of data, plots it, and returns a description.

    Args:
        data: A string containing data in CSV format.
    """
    ...


if __name__ == "__main__":
    # 5. Prepare the input data.
    sales_data = """Month,Sales
Jan,100
Feb,120
Mar,150
Apr,140
May,180
Jun,220
Jul,210
Aug,200
Sep,170
Oct,160
Nov,140
Dec,150
"""

    # 6. Run the agent task.
    result = analyze_data(data=sales_data)
    print(f"Agent's analysis: {result}")

    # 7. Optionally, save the plot to a file to verify it was created correctly.
    if os.path.exists("sales_plot.png"):
        os.remove("sales_plot.png")

    df = pd.read_csv(io.StringIO(sales_data))
    plt.figure(figsize=(8, 5))
    plt.bar(df["Month"], df["Sales"])
    plt.title("Monthly Sales")
    plt.xlabel("Month")
    plt.ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("sales_plot.png")
    print("\nSaved plot to sales_plot.png")
