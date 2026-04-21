"""Budget-controlled value rendering backed by reprobate.

reprobate's ``__init__`` eagerly auto-registers optional ext modules
(PIL, pandas, numpy, …), so importing it at module load would pull all
those packages into every ``import agex``. We defer it to the first
``render_value`` call.
"""


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
    import reprobate  # noqa: PLC0415

    return reprobate.render(value, budget=budget)
