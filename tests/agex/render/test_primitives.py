"""Unit tests for ``agex.render.primitives.is_dataframe``."""

import pandas as pd

from agex.render.primitives import is_dataframe


def test_dataframe_instance_is_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert is_dataframe(df) is True


def test_dataframe_class_is_not_a_dataframe():
    """``pd.DataFrame`` (the class) has class-level ``shape`` /
    ``columns`` attributes (a ``property`` descriptor, etc.) that
    pass ``hasattr`` but aren't the per-instance values.  Treating
    it as a DataFrame instance led downstream renderers to do
    ``cls.shape[0]`` and crash with ``TypeError: 'property' object
    is not subscriptable``."""
    assert is_dataframe(pd.DataFrame) is False


def test_arbitrary_class_is_not_a_dataframe():
    class Looksy:
        shape = (1, 2)
        columns = ["a", "b"]

    # The class itself fails the ``isinstance(value, type)`` filter
    # even though it has both attributes.
    assert is_dataframe(Looksy) is False


def test_series_is_not_a_dataframe():
    s = pd.Series([1, 2, 3])
    # Series has ``shape`` but not ``columns`` — already excluded.
    assert is_dataframe(s) is False


def test_collections_excluded():
    assert is_dataframe([1, 2, 3]) is False
    assert is_dataframe({"a": 1}) is False
    assert is_dataframe((1, 2)) is False
    assert is_dataframe({1, 2}) is False
