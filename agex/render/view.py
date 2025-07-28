from typing import Any, Literal, Union, overload

from ..agent import Agent
from ..state.versioned import Versioned
from .definitions import render_definitions
from .stream import StreamRenderer


@overload
def view(
    obj: Agent,
    *,
    full: bool = False,
) -> str: ...


@overload
def view(
    obj: Versioned,
    *,
    focus: Literal["recent", "full"] = "recent",
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
) -> Union[str, dict[str, Any]]: ...


def view(
    obj: Union[Agent, Versioned],
    *,
    focus: Literal["recent", "full"] = "recent",
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    full: bool = False,
) -> Union[str, dict[str, Any]]:
    """
    Provides a human-readable view of an agent's static capabilities or its
    current memory state.

    This is a debugging utility for quick, interactive inspection. For programmatic
    access to the event log, use `agex.events()`.

    - `view(agent)`: Shows the functions, classes, and modules registered with an
      agent.
    - `view(state)`: Shows a snapshot of the agent's memory.

    Args:
        obj: The Agent or Versioned state store to view.
        focus: For state views, the type of view to generate.
            "recent": A summary of state changes from the most recent execution.
            "full": The complete, raw key-value state at the current commit.
        full: For agent views, if True, shows all members regardless of visibility.
        model_name: The tokenizer model for the "recent" state view.
        max_tokens: The token budget for the "recent" state view.

    Returns:
        A string or dictionary depending on the view.

    Raises:
        ValueError: If the state has uncommitted live changes.
        TypeError: If an unsupported object type is provided.
    """
    if isinstance(obj, Agent):
        return render_definitions(obj, full=full)

    if isinstance(obj, Versioned):
        state = obj
        if state.live:
            raise ValueError("Cannot view state with uncommitted live changes.")

        if focus == "full":
            return {k: v for k, v in state.items() if not k.startswith("__")}

        if focus == "recent":
            if not state.current_commit:
                return ""

            # 1. Get the state changes from the most recent commit.
            state_changes = state.diffs()

            # 2. Render just the state stream.
            renderer = StreamRenderer(model_name=model_name)

            return renderer.render_state_stream(items=state_changes, budget=max_tokens)

        raise ValueError(f"Unknown view focus: {focus}")

    raise TypeError(f"view() not implemented for type '{type(obj).__name__}'")
