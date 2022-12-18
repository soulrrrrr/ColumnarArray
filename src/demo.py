import numpy as np
import _array as ar
import timeit

def make_matrices_np(size):

    mat1 = np.zeros([size,size])
    mat2 = np.zeros([size,size])
    mat3 = np.zeros([size,size])

    for it in range(size):
        for jt in range(size):
            mat1[it, jt] = it * size + jt + 1
            mat2[it, jt] = it * size + jt + 1
            mat3[it, jt] = 0

    return mat1, mat2, mat3

def make_matrices_ar(size):

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

def main():
    mat1, mat2, mat3, *_ = make_matrices_ar(1000)
    ns = dict(mat1=mat1, mat2=mat2)
    ar_time = timeit.timeit('ar_matmul(mat1, mat2)', 'from __main__ import ar_matmul', number=1, globals=ns)
    print(ar_time)
    mat1, mat2, mat3, *_ = make_matrices_np(1000)
    ns = dict(mat1=mat1, mat2=mat2)
    np_time = timeit.timeit('np_matmul(mat1, mat2)', 'from __main__ import np_matmul', number=1, globals=ns)
    print(np_time)

if __name__ == '__main__':
    main()
