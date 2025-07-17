from ..llm.core import ContentPart, TextPart
from ..state import State, Versioned
from .stream import StreamRenderer


class ContextRenderer:
    """
    Renders the current agent context (recent state changes and print output)
    into a list of multimodal ContentParts suitable for an LLM prompt,
    respecting a token budget.
    """

    def __init__(self, model_name: str):
        self._stream_renderer = StreamRenderer(model_name)

    def render(self, state: State, budget: int) -> list[ContentPart]:
        """
        Orchestrates the rendering of the current context.

        It prioritizes the most recent changes and gracefully degrades
        the level of detail to fit within the token budget.
        """
        all_parts: list[ContentPart] = []

        # 1. Get the changes from the most recent commit (if available).
        state_changes = {}
        if isinstance(state, Versioned):
            state_changes = state.diffs()

        stdout_content = state.get("__stdout__", [])

        # 2. Allocate independent budgets.
        state_budget = 0
        stdout_budget = 0
        if state_changes and stdout_content:
            state_budget = int(budget * 0.5)
            stdout_budget = int(budget * 0.5)
        elif state_changes:
            state_budget = budget
        elif stdout_content:
            stdout_budget = budget

        # 3. Render state stream (which is always text).
        if state_changes:
            rendered_state = self._stream_renderer.render_state_stream(
                items=state_changes, budget=state_budget
            )
            if rendered_state:
                all_parts.append(TextPart(text=rendered_state))

        # 4. Render stdout stream (which can be multimodal).
        if stdout_content:
            header = "Agent printed:\n"
            header_cost = self._stream_renderer.tokenizer.encode(header)
            item_budget = stdout_budget - len(header_cost)

            if item_budget > 0:
                stdout_parts = self._stream_renderer.render_item_stream(
                    items=stdout_content,
                    budget=item_budget,
                )
                if stdout_parts:
                    all_parts.append(TextPart(text=header))
                    all_parts.extend(stdout_parts)

        return all_parts
