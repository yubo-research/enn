import numpy as np
import pytest
from enn.turbo.types.appendable_array import AppendableArray


def test_appendable_array_basic():
    arr = AppendableArray(initial_capacity=2)
    assert len(arr) == 0
    assert arr.shape == (0, 0)

    # First append infers columns
    arr.append(np.array([1.0, 2.0]))
    assert len(arr) == 1
    assert arr.shape == (1, 2)
    np.testing.assert_array_equal(arr.view(), [[1.0, 2.0]])

    # Second append
    arr.append(np.array([[3.0, 4.0]]))
    assert len(arr) == 2
    assert arr.shape == (2, 2)
    np.testing.assert_array_equal(arr.view(), [[1.0, 2.0], [3.0, 4.0]])

    # Third append triggers growth (capacity 2 -> 4)
    arr.append(np.array([5.0, 6.0]))
    assert len(arr) == 3
    assert arr.shape == (3, 2)
    np.testing.assert_array_equal(arr.view(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_appendable_array_invalid_shape():
    arr = AppendableArray()
    arr.append(np.array([1.0, 2.0]))

    # Wrong number of columns
    with pytest.raises(ValueError, match="Expected 2 columns, got 3"):
        arr.append(np.array([1.0, 2.0, 3.0]))

    # Wrong number of rows in 2D input
    with pytest.raises(ValueError, match=r"Expected shape \(1, 2\), got \(2, 2\)"):
        arr.append(np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_appendable_array_view_is_no_copy():
    arr = AppendableArray()
    arr.append(np.array([1.0, 2.0]))
    v = arr.view()

    # Modify the view and check if the buffer changed (it should)
    v[0, 0] = 99.0
    np.testing.assert_array_equal(arr.view()[0, 0], 99.0)

    # Append more and check if view is still valid (it might be truncated, but the memory is shared)
    arr.append(np.array([3.0, 4.0]))
    # Note: v is still a view of the OLD buffer size or old buffer memory if growth happened.
    # But for the same buffer, it's shared.


def test_appendable_array_indexing():
    arr = AppendableArray()
    arr.append(np.array([1.0]))
    arr.append(np.array([2.0]))
    arr.append(np.array([3.0]))

    assert arr[0, 0] == 1.0
    np.testing.assert_array_equal(arr[1:], [[2.0], [3.0]])
    assert len(arr) == 3


def test_appendable_array_empty():
    arr = AppendableArray()
    assert arr.view().size == 0
    assert arr.shape == (0, 0)
