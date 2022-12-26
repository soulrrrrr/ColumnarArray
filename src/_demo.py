import numpy as np
import _array as ar
import timeit
import sys

# code from https://github.com/yungyuc/nsdhw_22au/blob/master/hw3/validate.py

def make_matrices_np(size, type):

    if type == 'i':
        mat1 = np.zeros([size,size], dtype='int')
        mat2 = np.zeros([size,size], dtype='int')
        mat3 = np.zeros([size,size], dtype='int')
    else:
        mat1 = np.zeros([size,size], dtype='float64')
        mat2 = np.zeros([size,size], dtype='float64')
        mat3 = np.zeros([size,size], dtype='float64')

    for it in range(size):
        for jt in range(size):
            mat1[it, jt] = it * size + jt + 1
            mat2[it, jt] = it * size + jt + 1
            mat3[it, jt] = 0

    return mat1, mat2, mat3

def make_matrices_ar(size, type):

    if type == 'i':
        mat1 = ar.zeros([size,size])
        mat2 = ar.zeros([size,size])
        mat3 = ar.zeros([size,size])
    else:
        mat1 = ar.zerosF([size,size])
        mat2 = ar.zerosF([size,size])
        mat3 = ar.zerosF([size,size])

    for it in range(size):
        for jt in range(size):
            mat1[it, jt] = it * size + jt + 1
            mat2[it, jt] = it * size + jt + 1
            mat3[it, jt] = 0

    return mat1, mat2, mat3

def np_matmul(mat1, mat2):
    mat = mat1 @ mat2
    return mat

def ar_matmul(mat1, mat2):
    mat = mat1 @ mat2
    return mat

def demo(size):
    mat1, mat2, mat3, *_ = make_matrices_ar(size, 'i')
    ns = dict(mat1=mat1, mat2=mat2)
    ar_time = timeit.timeit('ar_matmul(mat1, mat2)', 'from __main__ import ar_matmul', number=5, globals=ns)
    print(f"_array int time: {ar_time/5:.5f}s")
    mat1, mat2, mat3, *_ = make_matrices_np(size, 'i')
    ns = dict(mat1=mat1, mat2=mat2)
    np_time = timeit.timeit('np_matmul(mat1, mat2)', 'from __main__ import np_matmul', number=5, globals=ns)
    print(f"numpy int time: {np_time/5:.5f}s")

    mat1, mat2, mat3, *_ = make_matrices_ar(size, 'f')
    ns = dict(mat1=mat1, mat2=mat2)
    ar_time = timeit.timeit('ar_matmul(mat1, mat2)', 'from __main__ import ar_matmul', number=5, globals=ns)
    print(f"_array float time: {ar_time/5:.5f}s")
    mat1, mat2, mat3, *_ = make_matrices_np(size, 'f')
    ns = dict(mat1=mat1, mat2=mat2)
    np_time = timeit.timeit('np_matmul(mat1, mat2)', 'from __main__ import np_matmul', number=5, globals=ns)
    print(f"numpy float time: {np_time/5:.5f}s")

if __name__ == '__main__':
    demo(int(sys.argv[1]))
