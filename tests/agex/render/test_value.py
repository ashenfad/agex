from unittest.mock import patch

from agex.render.value import render_value


def test_render_primitives():
    assert render_value(123) == "123"
    assert render_value(True) == "True"
    assert render_value(None) == "None"


def test_render_string():
    result = render_value("hello")
    assert "hello" in result


def test_budget_respected():
    """Budget guarantee: output never exceeds the requested budget."""
    values = [
        42,
        "a" * 500,
        list(range(100)),
        {"key": "value", "nested": {"a": 1, "b": 2}},
        {f"k{i}": i for i in range(50)},
    ]
    for budget in [10, 50, 100, 500]:
        for val in values:
            result = render_value(val, budget=budget)
            assert (
                len(result) <= budget
            ), f"budget={budget}, len={len(result)}, result={result!r}"


def test_dataframe_token_budget_path():
    """When token_budget is set and value is a DataFrame, uses token-counted rendering."""
    sentinel = "DATAFRAME_RENDERED"

    with (
        patch("agex.render.primitives.is_dataframe", return_value=True),
        patch(
            "agex.render.primitives.render_dataframe_with_budget",
            return_value=sentinel,
        ),
    ):
        # This SHOULD take the DataFrame path (has token_budget)
        result_with_budget = render_value("fake_df", budget=200, token_budget=1024)
        assert result_with_budget == sentinel


def test_no_dataframe_path_without_token_budget():
    """Without token_budget, DataFrames go through reprobate like any other value."""
    result = render_value([1, 2, 3], budget=200)
    assert "1" in result


def test_non_dataframe_with_token_budget_uses_reprobate():
    """Non-DataFrame values use reprobate even when token_budget is set."""
    result = render_value([1, 2, 3], budget=200, token_budget=1024)
    assert "1" in result
    assert "2" in result
    assert "3" in result
