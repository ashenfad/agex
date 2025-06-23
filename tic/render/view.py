from typing import Any, Literal, overload

from tic.state.versioned import Versioned

from .stream import StreamRenderer


@overload
def view(
    state: Versioned,
    focus: Literal["recent"],
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
) -> str: ...


@overload
def view(state: Versioned, focus: Literal["full"]) -> dict[str, Any]: ...


@overload
def view(state: Versioned, focus: Literal["stdout"]) -> list[str]: ...


def view(
    state: Versioned,
    focus: Literal["recent", "full", "stdout"] = "recent",
    model_name: str = "gpt-4",
    max_tokens: int = 4096,
) -> str | list[str] | dict[str, Any]:
    """
    Provides a human-readable view of the agent's state.

    This function should be called between evaluations, when the ephemeral
    state is empty.

    Args:
        state: The Versioned state store to view.
        focus: The type of view to generate.
            "recent": A rendered string of state changes and stdout from the
                      most recent commit.
            "full": The full, raw key-value state at the current commit.
            "stdout": The full stdout log as a list of strings.
        model_name: The name of the tokenizer model to use for the "recent" view.
        max_tokens: The maximum number of tokens for the "recent" view.

    Returns:
        A string, dictionary, or list of strings depending on the focus.

    Raises:
        ValueError: If the ephemeral state has uncommitted changes.
    """
    if state.ephemeral.keys():
        raise ValueError("Cannot view state with uncommitted ephemeral changes.")

    if focus == "full":
        return {k: v for k, v in state.items() if not k.startswith("__")}

    if focus == "stdout":
        return state.get("__stdout__", [])

    if focus == "recent":
        if not state.current_commit:
            return ""

        # 1. Get the state changes from the most recent commit.
        state_changes = state.diffs()

        # 2. Render just the state stream.
        renderer = StreamRenderer(model_name=model_name)

        return renderer.render_state_stream(items=state_changes, budget=max_tokens)

    raise ValueError(f"Unknown view focus: {focus}")
