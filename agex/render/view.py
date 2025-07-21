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
    focus: Literal["recent", "full", "events"] = "recent",
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
) -> Union[str, list[str], dict[str, Any]]: ...


def view(
    obj: Union[Agent, Versioned],
    *,
    focus: Literal["recent", "full", "events"] = "recent",
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
    full: bool = False,
) -> Union[str, list[str], dict[str, Any], list[Any]]:
    """
    Provides a human-readable view of an agent's API or its state.

    - If an Agent is provided, it renders the agent's API definition.
    - If a Versioned state is provided, it renders a view of the state.

    Args:
        obj: The Agent or Versioned state store to view.
        focus: For state views, the type of view to generate.
            "recent": State changes from the most recent commit.
            "full": The full, raw key-value state at the current commit.
            "events": The full event log as a list of event objects.
        full: For agent views, if True, shows all members regardless of visibility.
        model_name: The tokenizer model for the "recent" state view.
        max_tokens: The token budget for the "recent" state view.

    Returns:
        A string, dictionary, or list of strings depending on the view.

    Raises:
        ValueError: If the state has uncommitted ephemeral changes.
        TypeError: If an unsupported object type is provided.
    """
    if isinstance(obj, Agent):
        return render_definitions(obj, full=full)

    if isinstance(obj, Versioned):
        state = obj
        if state.ephemeral:
            raise ValueError("Cannot view state with uncommitted ephemeral changes.")

        if focus == "full":
            return {k: v for k, v in state.items() if not k.startswith("__")}

        if focus == "events":
            from agex.state.log import get_events_from_log

            return get_events_from_log(state)

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
