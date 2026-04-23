import pandas as pd

from agex import Agent, connect_fs, connect_state
from agex.llm import Dummy
from tests.agex._emissions import make_response


def test_pandas_vfs_integration():
    """Test pandas reading/writing files in VFS via agent code."""

    # 1. Setup Agent with VFS and pandas
    responses = [
        make_response(
            thinking="I need to read the CSV, filter it, and save the result.",
            code="""
import pandas as pd

# Read from VFS using file handle (ensures patching works)
# with open("data/input.csv", "r") as f:
#     df = pd.read_csv(f)
df = pd.read_csv("data/input.csv")

# Process (filter for age > 30)
filtered_df = df[df["age"] > 30]

# Write back to VFS
# with open("data/output.csv", "w") as f:
filtered_df.to_csv("data/output.csv", index=False)
task_success("Processed data")
""",
        )
    ]

    agent = Agent(
        llm=Dummy(responses=responses),
        state=connect_state(type="versioned", storage="memory"),
        fs=connect_fs(type="virtual"),
    )

    # Register pandas with high visibility
    agent.module(pd, visibility="high")

    # 2. Setup initial data in VFS
    initial_csv = "name,age,city\nAlice,25,NY\nBob,35,SF\nCharlie,40,LA"
    fs = agent.fs()
    fs.write("data/input.csv", initial_csv.encode("utf-8"))

    # 3. Run agent task
    @agent.task
    def process_data() -> str:
        """Process the data."""
        pass

    result = process_data()

    # 4. Verify results
    assert result == "Processed data"

    # Check output file existence and content
    assert fs.exists("data/output.csv")
    output_content = fs.read("data/output.csv").decode("utf-8")

    # Bob and Charlie should be in output, Alice should be filtered out
    assert "Alice" not in output_content
    assert "Bob" in output_content
    assert "Charlie" in output_content

    # Verify metadata
    meta = fs.stat("data/output.csv")
    assert meta.size > 0
    assert meta.created_at is not None
