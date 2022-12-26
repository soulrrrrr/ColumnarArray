# Contiguous array

## Problem to Solve

Contiguous Array is basic for most of scientific calculations. Espicially in machine
learning area. I want to implement my version of it, and I hope it helps people
dealing with problems that need to use contigious array and matrix.

## Prospective Users

People wants lighter library for basic array and matrix operations.

## System Architecture

A class template :cpp:class:`!Array` is the contigious array. There are several methods
to initialize the array and use them. Each method will be exposed to Python by pybind11
wrapper.

At the moment I am planning, :cpp:class:`!Array` can only deal with 
structs have numeric types only.

## Compile library

Run `make` in src file to compile.
Run `make test` to run tests.
Run `make clean` to clean the build artifacts.
You may need to install Intel mkl library first.

## API Description

### Create array

```python
> import _array as ar
> a = ar.Array([1, 2, 3, 4, 5, 6])
> a.size
6
> a.dim
[6]
> b = ar.ArrayF([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
> b.size
6
> b.dim
[6]
```

### Use reshape to change to ndarray

```python
> a = a.reshape([2, 3])
> a.size
6
> a.dim
[2, 3]
```

### Matrix multiplication

```python
> b = ar.Array([1, 2, 3, 4, 5, 6])
> b = b.reshape([2, 3])
> a @ b
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: array 1's dim[1] should be equal to array 2's dim[0]
> b = b.reshape([3, 2])
> c = a @ b
> c.size
4
> c.dim
[2, 2]
> c.view()
[[ 22  28] 
 [ 49  64]] 

```
