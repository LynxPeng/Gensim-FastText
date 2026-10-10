import copy

import numpy as np
import pytest

from gensim_lab.array_checks import validate_array_roundtrip


def test_nested_hierarchical_softmax_arrays_roundtrip():
    values = np.empty(3, dtype=object)
    values[:] = [np.array([0, 1], dtype=np.uint8), np.array([], dtype=np.uint32),
                 np.array([3, 7, 10], dtype=np.uint32)]
    result = validate_array_roundtrip(values, copy.deepcopy(values))
    assert result["nested_object_entries"] == 3
    changed = copy.deepcopy(values)
    changed[2][1] = 8
    with pytest.raises(AssertionError, match="values changed"):
        validate_array_roundtrip(values, changed)
    changed = copy.deepcopy(values)
    changed[0] = changed[0].astype(np.uint32)
    with pytest.raises(AssertionError, match="dtype changed"):
        validate_array_roundtrip(values, changed)


def test_nonfinite_numeric_leaves_are_rejected():
    for value in [np.array([float("nan")]), np.array([float("inf")]),
                  np.array([np.array([float("inf")])], dtype=object)]:
        with pytest.raises(AssertionError):
            validate_array_roundtrip(value, copy.deepcopy(value))
