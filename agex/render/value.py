"""Budget-controlled value rendering backed by reprobate."""

import reprobate


def render_value(value, budget=2048, token_budget=None):
    """Render a Python value to a budget-constrained string.

    When token_budget is set and value is a DataFrame, uses iterative
    token-counted rendering for optimal tabular display. Otherwise
    delegates to reprobate.render().
    """
    if token_budget is not None:
        from .primitives import is_dataframe, render_dataframe_with_budget

        if is_dataframe(value):
            rendered = render_dataframe_with_budget(value, token_budget)
            return rendered[:budget] if len(rendered) > budget else rendered
    return reprobate.render(value, budget=budget)
