#!/usr/bin/env python3
import pytest
import colarray

def test_show():
    a = colarray.ColArray([[1, 3, 5], [2, 4, 6], [3, 6, 9]])
    a.show()