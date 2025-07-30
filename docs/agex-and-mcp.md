# A Note on Tooling (MCP) and the `agex` Philosophy

You may have noticed that the `agex` documentation emphasizes direct integration with Python libraries rather than compatibility with emerging "tool use" standards like the Machine-Callable Pool (MCP). This was a deliberate design choice, and this document explains the philosophy behind our code-native approach.

## The `agex` Premise: PyPI is the Ultimate Marketplace

The industry's move toward standardizing agent tools via MCP is an important effort to create a common language for agents to discover and use external, stateless services. It aims to build a new marketplace of capabilities.

`agex` is built on a different and more powerful premise:

> The richest, most mature, and most capable ecosystem of "tools" for a Python agent already exists. **It's called the Python Package Index (PyPI).**

Instead of asking developers to wrap their logic in new schemas for a new marketplace, `agex` enables agents to directly use the 500,000+ libraries that the Python community has spent decades building, testing, and perfecting.

## The Practical Difference: A Developer, Not a Form-Filler

This philosophical difference leads to a very practical difference in workflow.

**The Question:** "How can my agent use a third-party weather API?"

*   **The Standard "Tool" Answer:** "You need to find or create an MCP-compliant 'tool' that wraps the weather API. The agent can then be granted access to this tool and learn to call it by providing the correct JSON input."

*   **The `agex` Answer:** "Does the weather API have a Python client library?" If the answer is yes, the solution is immediate:

    ```python
    # 1. Install the existing library
    pip install official-weather-api-client

    # 2. Give the client to your agent
    import official_weather_api_client
    weather_client = official_weather_api_client.Client(api_key="...")
    agent.module(weather_client, name="weather")
    ```

The agent's role fundamentally changes. It is no longer a "form-filler" limited to the single, high-level function exposed by the tool. It is now a "developer" that can access the full API surface of the client library—all its methods, data classes, and enums—to compose a solution to the user's request in a single, efficient execution.

## The `agex` Advantage

This "code-native" approach has several key advantages:

*   **Power and Flexibility:** An agent can compose multiple low-level functions from a library (`get_historical_data`, `get_forecast`, `calculate_average`) in one turn to answer a complex question, something that would require multiple, high-latency round trips in a typical tool-based system.
*   **No Boilerplate:** You don't need to write or maintain custom JSON schemas or tool definitions. Instead, you can focus on curating the *ideal subset* of a library's API to expose to the agent, providing guidance and security without unnecessary boilerplate.
*   **Leverages the Entire Ecosystem:** It unlocks the long tail of the Python ecosystem, not just the small subset of libraries that have been explicitly wrapped as tools.

While one could wrap a Python library and expose it as an MCP tool, this often diminishes the library's power. It forces you to choose a few high-level functions to expose, hiding the rich, low-level functionality that enables agents to solve novel problems creatively.

The `agex` thesis is that agents are capable enough to work with the same powerful libraries that human developers do. Our goal is to provide the secure runtime environment that makes this powerful new paradigm possible.
