import pytest

from agex import Agent
from agex.agent.datatypes import MemberSpec
from agex.eval.user_errors import AgexAttributeError
from agex.llm.core import LLMResponse
from agex.llm.dummy_client import DummyLLMClient


class DatabaseConnection:
    """A mock database connection class to simulate a live, stateful object."""

    def __init__(self, db_name: str):
        self._db_name = db_name
        self._data = {"users": {1: "Alice", 2: "Bob"}}
        self.connection_id = "conn_123"

    def query(self, table: str, record_id: int) -> str:
        """Runs a 'query' on the mock database."""
        return self._data.get(table, {}).get(record_id, "Not Found")

    def _internal_method(self):
        """An internal method that should not be exposed."""
        return "internal"


def test_register_and_use_live_object():
    """End-to-end test that a live object can be registered and its methods/properties used."""
    db = DatabaseConnection("test_db")
    agent = Agent(primer="Test agent for live objects.", max_iterations=2)

    # Set up dummy LLM with the exact code we want to execute
    responses = [
        LLMResponse(
            thinking="I need to query the database and get the connection ID.",
            code='user = db.query("users", inputs.user_id)\nconn_id = db.connection_id\nexit_success((user, conn_id))',
        )
    ]
    agent.llm_client = DummyLLMClient(responses=responses)

    agent.module(
        db,
        name="db",
        configure={
            "connection_id": MemberSpec(visibility="high"),
            "query": MemberSpec(visibility="high"),
        },
    )

    @agent.task("Get a user from the database and return the connection ID.")
    def get_user_and_conn_id(user_id: int) -> tuple[str, str]:  # type: ignore[return-value]
        """
        Gets a user from the db and also returns the connection ID.

        Args:
            user_id: The ID of the user to retrieve

        Returns:
            A tuple containing the user name and connection ID
        """
        pass

    # Execute the task
    result = get_user_and_conn_id(1)

    assert result == ("Alice", "conn_123")


def test_live_object_state_safety():
    """Tests that assigning a bound method to a variable does not cause a pickle error."""
    db = DatabaseConnection("test_db")
    agent = Agent(primer="Test agent for state safety.", max_iterations=2)

    # Set up dummy LLM response
    responses = [
        LLMResponse(
            thinking="I'll assign the method to a variable and then call it.",
            code='query_method = db.query\nresult = query_method("users", inputs.user_id)\nexit_success(result)',
        )
    ]
    agent.llm_client = DummyLLMClient(responses=responses)

    agent.module(db, name="db", configure={"query": MemberSpec(visibility="high")})

    @agent.task("Assign a method to a variable and then call it.")
    def assign_method_to_var(user_id: int) -> str:  # type: ignore[return-value]
        """Assigns a method to a variable and then calls it."""
        pass

    # The key test is that this call completes without a PicklingError.
    result = assign_method_to_var(2)

    assert result == "Bob"


def test_register_instance_without_name_fails():
    """Tests that agent.module raises TypeError for an instance without a name."""
    db = DatabaseConnection("test_db")
    agent = Agent(primer="Test agent.")

    with pytest.raises(TypeError, match="The 'name' parameter is required"):
        agent.module(db)


def test_access_unexposed_member_fails():
    """Tests that accessing a member not included in the spec fails."""
    from agex.eval.core import evaluate_program
    from agex.state.kv import Memory
    from agex.state.namespaced import Namespaced
    from agex.state.versioned import Versioned

    db = DatabaseConnection("test_db")
    agent = Agent(primer="Test agent.")

    # We register the instance but do NOT expose the 'query' method.
    agent.module(db, name="db", exclude=["query"])

    # Set up state for direct evaluation
    versioned_state = Versioned(Memory())
    exec_state = Namespaced(versioned_state, namespace=agent.name)
    exec_state.set("__stdout__", [])

    # Test code that tries to access unexposed method
    code_to_test = 'result = db.query("users", 1)'

    with pytest.raises(
        AgexAttributeError, match="'db' object has no attribute 'query'"
    ):
        evaluate_program(code_to_test, agent, exec_state, 30.0)
