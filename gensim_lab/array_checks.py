"""Exact round-trip checks for numeric and nested Gensim arrays."""
from numbers import Number

import numpy as np


def _assert_equal_finite(before, after):
    if isinstance(before, np.ndarray):
        assert isinstance(after, np.ndarray), "Restored value is not an array"
        assert before.shape == after.shape, "Array shape changed"
        assert before.dtype == after.dtype, "Array dtype changed"
        if before.dtype.hasobject:
            for original, restored in zip(before.flat, after.flat):
                _assert_equal_finite(original, restored)
        else:
            assert np.array_equal(before, after), "Array values changed"
            if np.issubdtype(before.dtype, np.number):
                assert np.all(np.isfinite(before)), "Non-finite numeric array"
    elif isinstance(before, (list, tuple)):
        assert type(before) is type(after) and len(before) == len(after)
        for original, restored in zip(before, after):
            _assert_equal_finite(original, restored)
    else:
        assert before == after, "Nested scalar changed"
        if isinstance(before, Number):
            assert bool(np.isfinite(before)), "Non-finite numeric scalar"


def validate_array_roundtrip(before, after):
    """Check every element, including HS code/point arrays in object entries."""
    assert isinstance(before, np.ndarray)
    _assert_equal_finite(before, after)
    return {"shape": list(before.shape), "bytes": before.nbytes,
            "nested_object_entries": before.size if before.dtype.hasobject else 0}
