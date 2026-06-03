"""
Shallow, sampling-based validation for large data structures.
"""

import inspect
from dataclasses import is_dataclass
from typing import Any, get_args, get_origin

from pydantic import ConfigDict, TypeAdapter, ValidationError
from pydantic.errors import PydanticUserError

DEFAULT_SAMPLING_THRESHOLD = 100
DEFAULT_SAMPLE_SIZE = 10


# Strict configuration that prevents type coercion to catch type mismatches
STRICT_CONFIG = ConfigDict(arbitrary_types_allowed=True, strict=True)


def validate_with_sampling(value: Any, annotation: Any) -> Any:
    """
    Validates a value against a type annotation using Pydantic, but with
    sampling for large collections to avoid performance bottlenecks.

    Args:
        value: The value to validate.
        annotation: The type annotation (e.g., `list[int]`, `dict[str, float]`).

    Returns:
        The validated value. Pydantic may coerce types (e.g., str to int).

    Raises:
        ValidationError: If validation fails for the object or its samples.
    """
    # Sandbox-defined classes (sandtrap ``StClass``) are not Python types, so
    # neither ``isinstance`` nor Pydantic can validate against them (Pydantic
    # only warns and passes everything). A value built by ``SomeClass(...)`` in
    # the sandbox is an ``StInstance`` carrying the ``StClass`` that built it, so
    # validate by that identity. This only triggers for spawn return types
    # (host annotations are never ``StClass``). See roadmap/spawn-type-sharing.md.
    try:
        from sandtrap.wrappers import StClass, StInstance
    except Exception:  # pragma: no cover - sandtrap always present in practice
        StClass = StInstance = ()  # type: ignore[assignment]
    if isinstance(annotation, StClass):
        if isinstance(value, StInstance):
            st_class = object.__getattribute__(value, "_st_class")
            if st_class is annotation or (
                getattr(st_class, "_compiled_cls", None) is not None
                and getattr(st_class, "_compiled_cls", None)
                is getattr(annotation, "_compiled_cls", None)
            ):
                return value
        raise TypeError(
            f"Expected an instance of sandbox class "
            f"'{getattr(annotation, '_name', annotation)}', "
            f"got {type(value).__name__}"
        )

    # Peek validation for numpy arrays
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            # For a numpy array, we don't validate the whole array for performance.
            # Instead, we 'peek' at the first few items to validate their type.
            item_type = (
                annotation.item_type if hasattr(annotation, "item_type") else Any
            )
            adapter = TypeAdapter(list[item_type])
            sample = value[:DEFAULT_SAMPLE_SIZE]
            adapter.validate_python(sample)
            return value  # Return the original, un-truncated array
    except ImportError:
        # If numpy is not installed, we just skip this special handling.
        pass

    origin_type = get_origin(annotation)

    # For lists and tuples, apply sampling if they exceed the threshold
    if origin_type in (list, tuple) and isinstance(value, (list, tuple)):
        if len(value) > DEFAULT_SAMPLING_THRESHOLD:
            return _validate_sequence_sample(value, annotation)

    # For sets
    if origin_type is set and isinstance(value, set):
        if len(value) > DEFAULT_SAMPLING_THRESHOLD:
            return _validate_set_sample(value, annotation)

    # For dicts
    if origin_type is dict and isinstance(value, dict):
        if len(value) > DEFAULT_SAMPLING_THRESHOLD:
            return _validate_dict_sample(value, annotation)

    # For all other types, or collections below the threshold, validate normally.
    # Special case: for top-level standard dataclass annotations, do not coerce from dict.
    if inspect.isclass(annotation) and is_dataclass(annotation):
        # Check if value is already an instance of the expected dataclass
        if isinstance(value, annotation):
            return value
        # SPECIAL: Check if value is a dataclass with the same structure
        # (handles pickled dataclasses that have different identity but same structure)
        if is_dataclass(value):
            # Compare by name and fields
            if (
                type(value).__name__ == annotation.__name__
                and hasattr(value, "__dataclass_fields__")
                and hasattr(annotation, "__dataclass_fields__")
            ):
                value_fields = set(value.__dataclass_fields__.keys())
                annotation_fields = set(annotation.__dataclass_fields__.keys())
                if value_fields == annotation_fields:
                    # Same structure - accept it
                    return value
        # Block coercion of dict -> dataclass in strict mode
        raise TypeError(
            f"Expected instance of dataclass '{annotation.__name__}', got {type(value).__name__}"
        )
    try:
        adapter = TypeAdapter(annotation, config=STRICT_CONFIG)
    except PydanticUserError:
        # BaseModel, dataclass, or TypedDict as top-level annotation
        # cannot accept a config on the TypeAdapter. Fall back to no-config.
        adapter = TypeAdapter(annotation)
    try:
        return adapter.validate_python(value)
    except ValidationError:
        # Re-raise the original ValidationError - it already contains all the necessary information
        raise


def _validate_sequence_sample(sequence: list | tuple, annotation: Any) -> list | tuple:
    """
    Validates a sample of a large sequence (list or tuple).

    It validates the first `DEFAULT_SAMPLE_SIZE` and the last `DEFAULT_SAMPLE_SIZE`
    elements.
    """
    item_type = get_args(annotation)[0] if get_args(annotation) else Any
    adapter = TypeAdapter(list[item_type], config=STRICT_CONFIG)

    head = sequence[:DEFAULT_SAMPLE_SIZE]
    tail = sequence[-DEFAULT_SAMPLE_SIZE:]

    # Validate the head and tail samples.
    # Pydantic will return a new list with potentially coerced values.
    validated_head = adapter.validate_python(head)
    validated_tail = adapter.validate_python(tail)

    # Important: Return a new sequence with the validated (and possibly
    # type-coerced) head and tail, stitched back together with the
    # un-validated middle.
    # This preserves the original data while ensuring the samples are correct.
    # It also means we pass the *partially* validated data to the agent.
    original_type = type(sequence)
    middle = list(sequence[DEFAULT_SAMPLE_SIZE:-DEFAULT_SAMPLE_SIZE])
    return original_type(validated_head + middle + validated_tail)


def _validate_set_sample(value: set, annotation: Any) -> set:
    """
    Validates a sample of a large set.
    Since sets are unordered, this takes the first `DEFAULT_SAMPLE_SIZE` elements
    after converting the set to a list.
    """
    item_type = get_args(annotation)[0] if get_args(annotation) else Any
    adapter = TypeAdapter(list[item_type], config=STRICT_CONFIG)

    # Convert set to list to get a sample
    value_list = list(value)
    sample = value_list[:DEFAULT_SAMPLE_SIZE]
    validated_sample = adapter.validate_python(sample)

    # Return a new set with the validated sample and the rest of the items.
    # This is not perfectly efficient, but it's the only way to be sure.
    return set(validated_sample) | set(value_list[DEFAULT_SAMPLE_SIZE:])


def _validate_dict_sample(value: dict, annotation: Any) -> dict:
    """
    Validates a sample of a large dictionary.
    It performs head/tail sampling on the dictionary's items.
    """
    key_type, value_type = (
        get_args(annotation) if len(get_args(annotation)) == 2 else (Any, Any)
    )
    adapter = TypeAdapter(list[tuple[key_type, value_type]], config=STRICT_CONFIG)

    item_list = list(value.items())
    head = item_list[:DEFAULT_SAMPLE_SIZE]
    tail = item_list[-DEFAULT_SAMPLE_SIZE:]

    validated_head = adapter.validate_python(head)
    validated_tail = adapter.validate_python(tail)

    # Reconstruct the dictionary
    middle_items = item_list[DEFAULT_SAMPLE_SIZE:-DEFAULT_SAMPLE_SIZE]
    return dict(validated_head + middle_items + validated_tail)
