# Registration Helpers

To make it easier to get started with common libraries, `agex` provides a set of registration helpers. These helpers are simple functions that pre-register popular libraries like NumPy, pandas, and the Python standard library with sensible defaults.

## Philosophy

The core idea behind these helpers is to:

1.  **Save Time**: Provide a one-line way to give agents access to powerful, well-known libraries.
2.  **Use Sensible Defaults**: The helpers register modules with `visibility="low"`, which means they are available to the agent but don't clutter the agent's limited context window with detailed documentation. LLMs are already extensively trained on these libraries, so they don't need to see the full docstrings.
3.  **Demonstrate Best Practices**: The helpers serve as a practical example of how to use `agent.module` and `agent.cls` to create your own registration patterns for your internal tools and libraries.
4.  **Promote Security**: They exclude potentially dangerous functions (like `os.system` or file I/O operations in data libraries) that are not suitable for a sandboxed agent environment.

## Usage

To use a helper, simply import it and pass your agent instance to it.

```python
from agex import Agent
from agex.helpers import register_numpy, register_pandas, register_stdlib

# Create an agent
data_analyst = Agent(name="data_analyst")

# Register libraries with one line each
register_numpy(data_analyst)
register_pandas(data_analyst)
register_stdlib(data_analyst)

# The agent now has access to these libraries
@data_analyst.task
def analyze(data: list) -> float:
    """Calculate the mean of the data using pandas."""
    pass
```

## Available Helpers

### `register_stdlib(agent)`

Registers a curated list of safe and useful modules from the Python standard library.

-   **Mathematical**: `math`, `random` (with state-setting functions excluded), `statistics`, `decimal`, `fractions`.
-   **Utilities**: `collections`, `datetime` (including its classes), `uuid`.
-   **Text Processing**: `re`, `string`, `textwrap`.
-   **Data Encoding**: `json`, `base64`, `hashlib`.
-   All modules are registered with `visibility="low"`.

### `register_numpy(agent)`

Registers the `numpy` library for numerical computing.

-   Registers the core `numpy` module.
-   Registers useful sub-modules: `numpy.random`, `numpy.linalg`, `numpy.fft`, and `numpy.ma`.
-   Excludes potentially unsafe functions like `load`, `save`, and file I/O operations.
-   All modules are registered with `visibility="low"`.

### `register_pandas(agent)`

Registers the `pandas` library for data analysis.

-   Registers the core `pandas` module and the `pandas.api.types` submodule.
-   Explicitly registers accessor classes like `DatetimeProperties` (`.dt`), `StringMethods` (`.str`), and `Rolling` (`.rolling`) to ensure they are available to the agent.
-   Excludes all `read_*` and `to_*` functions to prevent file system access, aligning with a secure-by-default posture.
-   All modules and classes are registered with `visibility="low"`.

### `register_plotly(agent)`

Registers the `plotly` library for creating interactive visualizations.

-   Registers `plotly.express` and `plotly.graph_objects` for creating plots.
-   Registers `plotly.tools`, `plotly.colors`, and `plotly.figure_factory` for advanced plotting utilities.
-   Specifically registers the `go.Figure` class to ensure methods like `add_scatter` and `update_layout` are available.
-   Excludes functions related to writing files or showing plots directly (e.g., `write_image`, `show`), as these are side-effects that should be handled by the user's code, not the agent's.
-   All modules and classes are registered with `visibility="low"`. 