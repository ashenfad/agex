"""Integration tests for json and csv modules with VirtualFS."""

from agex import Agent, connect_fs, connect_state
from agex.llm import Dummy
from tests.agex._emissions import make_response


def test_json_vfs_integration():
    """Test json module reading/writing files in VFS via agent code."""
    responses = [
        make_response(
            thinking="I'll read the JSON, modify it, and write it back.",
            code="""import json

# Read JSON from VFS
with open("config.json", "r") as f:
    data = json.load(f)

# Modify the data
data["processed"] = True
data["count"] = data.get("count", 0) + 1

# Write back to VFS
with open("config.json", "w") as f:
    json.dump(data, f, indent=2)

task_success(f"Updated count to {data['count']}")
""",
        )
    ]

    agent = Agent(
        llm=Dummy(responses=responses),
        state=connect_state(type="versioned", storage="memory"),
        fs=connect_fs(type="virtual"),
    )

    # Setup initial JSON in VFS
    import json

    initial_data = {"name": "test", "count": 5, "enabled": True}
    fs = agent.fs()
    fs.write("config.json", json.dumps(initial_data).encode("utf-8"))

    # Run agent task
    @agent.task
    def process_config() -> str:
        """Process the config."""
        pass

    result = process_config()

    # Verify results
    assert result == "Updated count to 6"

    # Check output file content
    assert fs.exists("config.json")
    output_content = fs.read("config.json").decode("utf-8")
    output_data = json.loads(output_content)

    assert output_data["name"] == "test"
    assert output_data["count"] == 6
    assert output_data["enabled"] is True
    assert output_data["processed"] is True


def test_csv_vfs_integration():
    """Test csv module reading/writing files in VFS via agent code."""
    responses = [
        make_response(
            thinking="I'll read the CSV, filter rows, and write output.",
            code="""import csv

# Read CSV from VFS
with open("input.csv", "r") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter for active users only
active_users = [row for row in rows if row["status"] == "active"]

# Write filtered data to output
with open("output.csv", "w") as f:
    if active_users:
        writer = csv.DictWriter(f, fieldnames=active_users[0].keys())
        writer.writeheader()
        writer.writerows(active_users)

task_success(f"Filtered {len(active_users)} active users")
""",
        )
    ]

    agent = Agent(
        llm=Dummy(responses=responses),
        state=connect_state(type="versioned", storage="memory"),
        fs=connect_fs(type="virtual"),
    )

    # Setup initial CSV in VFS
    initial_csv = """name,status,score
Alice,active,95
Bob,inactive,80
Charlie,active,88
Diana,inactive,92"""

    fs = agent.fs()
    fs.write("input.csv", initial_csv.encode("utf-8"))

    # Run agent task
    @agent.task
    def filter_users() -> str:
        """Filter the users."""
        pass

    result = filter_users()

    # Verify results
    assert result == "Filtered 2 active users"

    # Check output file content
    assert fs.exists("output.csv")
    output_content = fs.read("output.csv").decode("utf-8")

    # Verify only active users in output
    assert "Alice" in output_content
    assert "Charlie" in output_content
    assert "Bob" not in output_content
    assert "Diana" not in output_content


def test_json_loads_dumps():
    """Test that json.loads/dumps work without file operations."""
    responses = [
        make_response(
            thinking="I'll use json.loads and json.dumps for string serialization.",
            code="""import json

# Parse JSON string
json_str = '{"x": 10, "y": 20}'
data = json.loads(json_str)

# Modify and serialize back
data["sum"] = data["x"] + data["y"]
result = json.dumps(data)

task_success(result)
""",
        )
    ]

    agent = Agent(
        llm=Dummy(responses=responses),
        state=connect_state(type="live", storage="memory"),
        fs=connect_fs(type="virtual"),
    )

    @agent.task
    def json_ops() -> str:
        """JSON operations."""
        pass

    result = json_ops()

    # Verify result is valid JSON with expected data
    import json

    data = json.loads(result)
    assert data["x"] == 10
    assert data["y"] == 20
    assert data["sum"] == 30


# Note: pathlib test removed - pathlib.Path needs additional VFS integration
# beyond current scope. Path objects work with open() but Path.read_text() etc
# bypass VFS wrappers and would need their own monkey-patching.
