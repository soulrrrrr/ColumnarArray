import pytest
import _array
import numpy as np
from _demo import demo

def test_init():
    a = _array.Array([1, 2, 4, 5])
    assert a.size == 4

def test_zeros():
    a = _array.Array([1, 2, 4, 5])
    b = _array.zeros([3, 3])
    assert b.size == 9

def test_get():
    b = _array.Array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    assert b[1, 0] == b[3]
    with pytest.raises(ValueError):
        b[0, 0, 0] = 0

def test_set():
    a = _array.Array([1, 2, 4, 5])
    b = _array.Array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    assert a[0] == 1
    a[0] = 3
    assert a[0] == 3
    assert b[1, 2] == 6
    b[5] = 7
    assert b[5] == 7

def test_add():
    a = _array.Array([i for i in range(9)]).reshape([3, 3])
    b = _array.ones([3, 3])
    c = _array.Array([i for i in range(1, 10)]).reshape([3, 3])
    assert a + b == c

def test_sub():
    a = _array.Array([i for i in range(1, 10)]).reshape([3, 3])
    b = _array.ones([3, 3])
    c = _array.Array([i for i in range(9)]).reshape([3, 3])
    assert a - b == c

def test_mul():
    a = _array.Array([i for i in range(1, 10)]).reshape([3, 3])
    b = _array.Array([i for i in range(1, 10)]).reshape([3, 3])
    c = _array.Array([i*i for i in range(1, 10)]).reshape([3, 3])
    assert a * b == c

def test_matmul():
    a = _array.Array([i for i in range(1, 7)]).reshape([2, 3])
    b = _array.Array([i for i in range(7, 19)]).reshape([3, 4])
    c = _array.Array([74, 80, 86, 92, 173, 188, 203, 218]).reshape([2, 4])
    assert a @ b == c

def test_float_init():
    a = _array.ArrayF([1.1, 2.2, 4.4, 5.5])
    assert a.size == 4

def test_float_zeros():
    a = _array.ArrayF([1.1, 2.2, 4.4, 5.5])
    b = _array.zerosF([3, 3])
    assert b.size == 9

def test_float_get():
    b = _array.ArrayF([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    assert b[1, 0] == b[3]
    with pytest.raises(ValueError):
        b[0, 0, 0] = 0

def test_float_set():
    a = _array.ArrayF([1, 2, 4, 5])
    b = _array.ArrayF([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    assert a[0] == 1
    a[0] = 3
    assert a[0] == 3
    assert b[1, 2] == 6
    b[5] = 7
    assert b[5] == 7

def test_float_add():
    a = _array.ArrayF([i for i in range(9)]).reshape([3, 3])
    b = _array.onesF([3, 3])
    c = _array.ArrayF([i for i in range(1, 10)]).reshape([3, 3])
    assert a + b == c

def test_float_sub():
    a = _array.ArrayF([i for i in range(1, 10)]).reshape([3, 3])
    b = _array.onesF([3, 3])
    c = _array.ArrayF([i for i in range(9)]).reshape([3, 3])
    assert a - b == c

def test_float_mul():
    a = _array.ArrayF([i for i in range(1, 10)]).reshape([3, 3])
    b = _array.ArrayF([i for i in range(1, 10)]).reshape([3, 3])
    c = _array.ArrayF([i*i for i in range(1, 10)]).reshape([3, 3])
    assert a * b == c

def test_float_matmul():
    a = _array.ArrayF([i for i in range(1, 7)]).reshape([2, 3])
    b = _array.ArrayF([i for i in range(7, 19)]).reshape([3, 4])
    c = _array.ArrayF([74, 80, 86, 92, 173, 188, 203, 218]).reshape([2, 4])
    assert a @ b == c
    d = c
    assert c == d

if __name__ == '__main__':
    demo()


