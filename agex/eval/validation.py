"""
Shallow, sampling-based validation for large data structures.
"""

from typing import Any, get_args, get_origin

from pydantic import ConfigDict, TypeAdapter, ValidationError

DEFAULT_SAMPLING_THRESHOLD = 100
DEFAULT_SAMPLE_SIZE = 10


# Pydantic configuration for handling arbitrary types like numpy arrays
NUMPY_AWARE_CONFIG = ConfigDict(arbitrary_types_allowed=True)


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
    try:
        adapter = TypeAdapter(annotation, config=NUMPY_AWARE_CONFIG)
        return adapter.validate_python(value)
    except ValidationError as e:
        # Re-raise with more context about what was being validated.
        raise ValidationError(f"Validation failed for type '{annotation}':\n{e}") from e


def _validate_sequence_sample(sequence: list | tuple, annotation: Any) -> list | tuple:
    """
    Validates a sample of a large sequence (list or tuple).

    It validates the first `DEFAULT_SAMPLE_SIZE` and the last `DEFAULT_SAMPLE_SIZE`
    elements.
    """
    item_type = get_args(annotation)[0] if get_args(annotation) else Any
    adapter = TypeAdapter(list[item_type], config=NUMPY_AWARE_CONFIG)

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
    adapter = TypeAdapter(list[item_type], config=NUMPY_AWARE_CONFIG)

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
    adapter = TypeAdapter(list[tuple[key_type, value_type]], config=NUMPY_AWARE_CONFIG)

    item_list = list(value.items())
    head = item_list[:DEFAULT_SAMPLE_SIZE]
    tail = item_list[-DEFAULT_SAMPLE_SIZE:]

    validated_head = adapter.validate_python(head)
    validated_tail = adapter.validate_python(tail)

    # Reconstruct the dictionary
    middle_items = item_list[DEFAULT_SAMPLE_SIZE:-DEFAULT_SAMPLE_SIZE]
    return dict(validated_head + middle_items + validated_tail)
