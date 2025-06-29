from ..state import State, Versioned
from .stream import StreamRenderer


class ContextRenderer:
    """
    Renders the current agent context (recent state changes and print output)
    into a string suitable for an LLM prompt, respecting a token budget.
    """

    def __init__(self, model_name: str):
        self._stream_renderer = StreamRenderer(model_name)

    def render(self, state: State, budget: int) -> str:
        """
        Orchestrates the rendering of the current context.

        It prioritizes the most recent changes and gracefully degrades
        the level of detail to fit within the token budget.
        """
        # 1. Get the changes from the most recent commit (if available).
        state_changes = {}
        if isinstance(state, Versioned):
            state_changes = state.diffs()
        # For Namespaced state, there are no diffs since we don't snapshot during sub-agent execution

        stdout_content = state.get("__stdout__", [])

        # 2. Allocate independent budgets.
        state_budget = 0
        stdout_budget = 0
        if state_changes and stdout_content:
            state_budget = int(budget * 0.6)
            stdout_budget = int(budget * 0.4)
        elif state_changes:
            state_budget = budget
        elif stdout_content:
            stdout_budget = budget

        # 3. Render each stream independently.
        rendered_stdout = self._stream_renderer.render_item_stream(
            items=stdout_content,
            budget=stdout_budget,
            header="Agent printed:\n",
        )

        rendered_state = self._stream_renderer.render_state_stream(
            items=state_changes, budget=state_budget
        )

        # 4. Combine the rendered outputs.
        return "\n".join(filter(None, [rendered_state, rendered_stdout]))
