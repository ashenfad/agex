"""Tests for pickle safety checking."""

import threading
from types import ModuleType

import pytest

from agex.eval.error import EvalError
from agex.eval.objects import AgexModule
from agex.eval.safe import check_assignment_safety


def test_safe_atomic_types():
    """Test that basic atomic types pass through safely."""
    safe_values = [42, 3.14, "hello", b"bytes", True, None, 1 + 2j, range(5)]

    for value in safe_values:
        result = check_assignment_safety(value)
        assert result is value  # Should return the exact same object


def test_module_conversion():
    """Test that ModuleType objects are converted to AgexModule."""
    module = ModuleType("test_module")
    result = check_assignment_safety(module)

    assert isinstance(result, AgexModule)
    assert result.name == "test_module"


def test_unpickleable_objects_fail():
    """Test that unpickleable objects raise EvalError."""
    # Test objects that don't have reliable pickle methods
    with pytest.raises(EvalError, match="Cannot assign unpickleable"):
        check_assignment_safety(threading.Lock())

    with pytest.raises(EvalError, match="Cannot assign unpickleable"):
        check_assignment_safety(threading.Thread(target=lambda: None))

    # Test file object (has __reduce__ but not reliable pickle methods)
    file_obj = open(__file__, "r")
    try:
        with pytest.raises(EvalError, match="Cannot assign unpickleable"):
            check_assignment_safety(file_obj)
    finally:
        file_obj.close()


def test_collections_with_bad_contents():
    """Test that collections containing unpickleable objects fail."""
    lock = threading.Lock()

    bad_collections = [
        [1, 2, lock],
        (1, 2, lock),
        {1, 2, lock},
        {"key": lock},
    ]

    for collection in bad_collections:
        with pytest.raises(EvalError, match="Cannot assign unpickleable"):
            check_assignment_safety(collection)


def test_collections_with_safe_contents():
    """Test that collections with safe contents pass through."""
    safe_collections = [
        [1, 2, 3],
        (1, 2, 3),
        {1, 2, 3},
        {"a": 1, "b": 2},
    ]

    for collection in safe_collections:
        result = check_assignment_safety(collection)
        assert result == collection


def test_dataclass_objects():
    """Test that dataclass objects are considered safe."""
    from dataclasses import dataclass

    @dataclass
    class Point:
        x: int
        y: int

    point = Point(1, 2)
    result = check_assignment_safety(point)
    assert result is point


def test_custom_pickle_support():
    """Test that objects with pickle dunders are considered safe."""

    class CustomPickleClass:
        def __init__(self, value):
            self.value = value

        def __getstate__(self):
            return {"value": self.value}

        def __setstate__(self, state):
            self.value = state["value"]

    obj = CustomPickleClass(42)
    result = check_assignment_safety(obj)
    assert result is obj


def test_numpy_objects():
    """Test that numpy objects are considered safe."""
    try:
        import numpy as np

        arr = np.array([1, 2, 3])
        result = check_assignment_safety(arr)
        assert result is arr
    except ImportError:
        pytest.skip("NumPy not available")


def test_iterator_helpful_errors():
    """Test that common iterator types give helpful error messages."""
    test_cases = [
        (
            {"a": 1}.keys(),
            "Cannot assign dict_keys object. Use list(dict.keys()) to convert to a list.",
        ),
        (
            {"a": 1}.values(),
            "Cannot assign dict_values object. Use list(dict.values()) to convert to a list.",
        ),
        (
            {"a": 1}.items(),
            "Cannot assign dict_items object. Use list(dict.items()) to convert to a list.",
        ),
        (
            map(str, [1, 2, 3]),
            "Cannot assign map object. Use list(map(...)) to convert to a list.",
        ),
        (
            filter(bool, [1, 0, 2]),
            "Cannot assign filter object. Use list(filter(...)) to convert to a list.",
        ),
        (
            enumerate([1, 2, 3]),
            "Cannot assign enumerate object. Use list(enumerate(...)) to convert to a list.",
        ),
    ]

    for obj, expected_message in test_cases:
        with pytest.raises(
            EvalError,
            match=expected_message.replace("(", r"\(")
            .replace(")", r"\)")
            .replace("...", r"\.\.\."),
        ):
            check_assignment_safety(obj)


def test_file_objects_with_reduce():
    """Test that file objects with __reduce__ still fail (fallback to pickle test)."""
    file_obj = open(__file__, "r")
    try:
        # File objects have __reduce__ but still aren't pickleable
        # The safety checker should fall back to the full pickle test
        with pytest.raises(EvalError, match="Cannot assign unpickleable"):
            check_assignment_safety(file_obj)
    finally:
        file_obj.close()
