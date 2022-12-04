import pytest
import _array

def test_init():
    a = _array.Array([1, 2, 4, 5])
    assert a.size == 4