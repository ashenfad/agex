# Agentic Vision: An Internal Reasoning Tool

This document outlines `agex`'s unique approach to integrating vision capabilities directly into an agent's reasoning loop. Unlike traditional multimodal models that treat images as user inputs or final outputs, `agex` empowers agents to *proactively* generate and "view" images as an intermediate step in their problem-solving process.

## A Philosophical Shift

In most frameworks, vision works in one of two ways:
1.  **User-to-Agent:** The user provides an image, and the agent describes or analyzes it.
2.  **Agent-to-User:** The agent generates an image as a final artifact to be displayed in a UI.

`agex` introduces a third paradigm: **Agent-to-Agent**. The agent can generate a visualization (e.g., a data plot) and immediately consume it as input for its subsequent thought process. This creates a powerful, iterative workflow where the agent can reason over data, visualize it to find patterns, and then reason over the visualization itself.

## The `view_image` Built-in

The core of this functionality is the `view_image(image_obj: Any)` built-in function.

### Dynamic Availability
`view_image` is not a standard registered function. It is a special, dynamic capability whose documentation is injected into the system primer *only* when a compatible image-handling library is registered with the agent. The supported libraries are:
- `matplotlib`
- `numpy`
- `PIL` (Pillow)

This ensures that the agent is only aware of the `view_image` function when it actually has the tools to create an image to view.

### Required Usage Pattern
To correctly use this feature, the agent must generate an image and then "pause" to allow the system to render it for the next turn. This is enforced by a specific calling sequence:

1.  Call a tool that generates an image object.
2.  Pass that object to `view_image()`.
3.  Immediately call `task_continue()` to end the current turn.

**Example Agent Action:**
```python
# First, agent calls a tool that creates a plot and returns the figure object
fig = create_my_plot(df)
# Then, agent uses view_image and task_continue to see it next turn
view_image(fig)
task_continue()
```
In the following turn, the image will be available in the agent's context, allowing it to perform analysis on the visualization it just created.

## Future Work
- [ ] Create detailed examples demonstrating this workflow (e.g., using `pandas` and `matplotlib`).
- [ ] Update the README and other high-level documentation to showcase this unique capability.

## Future Work: Auto-Viewing for Efficiency

Currently, an agent must take two turns to see an image passed as input:
1.  **Turn 1:** The agent calls `view_image(inputs.my_image)` and `task_continue()`.
2.  **Turn 2:** The agent sees the image and performs the analysis.

This is safe and explicit, but it's inefficient for simple tasks where the agent's only goal is to analyze a single, provided image (e.g., "count the cats in this picture").

To optimize this, we can introduce an opt-in mechanism that allows a developer to signal that an image argument should be automatically viewed by the agent on the first turn.

### Proposed Solution: `Annotated` for Explicit Opt-In

The most flexible and Pythonic solution is to use `typing.Annotated`. We will introduce a simple marker type, `Viewable`, that developers can use to annotate function arguments.

**Example:**
```python
from typing import Annotated
from PIL import Image
from agex import Viewable # This marker will be created

@my_agent.task
def count_the_cats(
    image_to_analyze: Annotated[Image.Image, Viewable]
) -> int:
    """Counts the number of cats in the provided image."""
    # With this annotation, the agent will see the image on turn 1
    # and can immediately respond, saving an entire iteration.
```

### Implementation Plan
- [ ] **Define Marker:** Create a simple `Viewable = type("Viewable", (), {})` in `agex/agent/datatypes.py`.
- [ ] **Detect Annotation:** In `agex/agent/task.py`, modify `_create_task_wrapper` to inspect task parameters for the `Annotated[..., Viewable]` signature.
- [ ] **Inject `view_image` Call:** In `agex/agent/loop.py`, update `_run_task_loop` and `_build_task_message` to accept the list of viewable parameter names. If the list is not empty, prepend `view_image(inputs.<param_name>)` calls to the initial task message sent to the agent.
- [ ] **Documentation**: Update the official documentation to explain this advanced, opt-in feature. 