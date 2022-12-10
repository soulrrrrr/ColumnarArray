import pytest
import _array
import numpy as np

def test_init():
    a = _array.Array([1, 2, 4, 5])
    assert a.size == 4

def test_zeros():
    a = _array.Array([1, 2, 4, 5])
    b = _array.zeros([3, 3])
    assert b.size == 9

def test_get():
    a = _array.Array([1, 2, 4, 5])
    b = _array.zeros([3, 3])
    assert b[0] == 0
    assert b[0, 0] == 0
    with pytest.raises(ValueError):
        b[0, 0, 0] = 0

def test_set():
    a = _array.Array([1, 2, 4, 5])
    b = _array.zeros([3, 3])
    assert a[0] == 1
    a[0] = 3
    assert a[0] == 3
    assert b[1, 2] == 0
    b[5] = 7
    assert b[5] == 7

def test_add():
    a = _array.Array([i for i in range(9)]).arange([3, 3])
    b = _array.ones([3, 3])
    c = _array.Array([i for i in range(1, 10)]).arange([3, 3])
    assert a + b == c

def test_sub():
    a = _array.Array([i for i in range(1, 10)]).arange([3, 3])
    b = _array.ones([3, 3])
    c = _array.Array([i for i in range(9)]).arange([3, 3])
    assert a - b == c

def test_mul():
    a = _array.Array([i for i in range(1, 10)]).arange([3, 3])
    b = _array.Array([i for i in range(1, 10)]).arange([3, 3])
    c = _array.Array([i*i for i in range(1, 10)]).arange([3, 3])
    assert a * b == c