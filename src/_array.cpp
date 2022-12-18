#include <iostream>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <iomanip>
#include <algorithm>
#include <numeric>
#include <mkl.h>
#include <type_traits>
using std::vector;

namespace py = pybind11;

void check_dim(vector<size_t> v1, vector<size_t> v2)
{
    if (v1 != v2)
        throw std::range_error("the dimension of 2 array should be same");
}

template<typename T>
class Array {
public:
    Array(): m_size(0) {}
    Array(int n): m_data(vector<T>(n, 0)), m_size(n), m_dim(vector<size_t>(1, n)) {}
    Array(vector<T> v): m_data(v), m_size(v.size()), m_dim(vector<size_t>(1, v.size())) {}
    Array(vector<T> v, vector<size_t> dim): m_data(v), m_size(v.size()), m_dim(dim) {}

    Array operator+(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < (int)m_size; i++)
            ret.m_data[i] += v.m_data[i];
        return ret;
    }

    Array operator-(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < (int)m_size; i++)
            ret.m_data[i] -= v.m_data[i];
        return ret;
    }

    Array operator*(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < (int)m_size; i++)
            ret.m_data[i] *= v.m_data[i];
        return ret;
    }

    bool operator==(Array& v)
    {
        if (m_size != v.m_size)
            return false;
        if (m_dim != v.m_dim)
            return false;
        if (m_data != v.m_data)
            return false;
        return true;
    }

    Array reshape(const vector<size_t>& v)
    {
        size_t size = 1;
        for (size_t i : v)
            size *= i;
        if (m_size != size)
            throw std::range_error("the size of 2 arrays are not same");

        m_dim.clear();
        for (size_t i : v)
            m_dim.push_back(i);

        return *this;
    }

public:
    size_t m_size = 0;
    vector<size_t> m_dim = {};
    vector<T> m_data = {};
};

template<typename T>
Array<T> zeros(vector<size_t> dim)
{
    size_t size = 1;
    for (size_t d : dim)
        size *= d;
    return Array(vector<T>(size, 0.0), dim);
}

template<typename T>
Array<T> ones(vector<size_t> dim)
{
    size_t size = 1;
    for (size_t d : dim)
        size *= d;
    return Array(vector<T>(size, 1.0), dim);
}

Array<double> matmul(Array<double> &mat1, Array<double> &mat2)
{

    if (mat1.m_dim.size() != 2 || mat2.m_dim.size() != 2)
        throw std::range_error("2 array should all be 2d array");
    if (mat1.m_dim[1] != mat2.m_dim[0])
        throw std::range_error("array 1's dim[1] should be equal to array 2's dim[0]");
    Array<double> ret(mat1.m_dim[0]*mat2.m_dim[1]);
    ret = ret.reshape({mat1.m_dim[0], mat2.m_dim[1]});
    // int n = a1.m_dim[0], m = a2.m_dim[0], l = a2.m_dim[1];
    // for (int i = 0; i < n; i++)
    // {
    //     for (int k = 0; k < l; k++)
    //     {
    //         int tmp = 0;
    //         for (int j = 0; j < m; j++)
    //         {
    //             tmp += (a1.m_data[i * m + j] * a2.m_data[j * l + k]);
    //         }
    //         ret.m_data[i * l + k] = tmp;
    //     }
    // }

    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
        , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
        , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
        , mat1.m_dim[0] /* const MKL_INT m */
        , mat2.m_dim[1] /* const MKL_INT n */
        , mat1.m_dim[1] /* const MKL_INT k */
        , 1.0 /* const double alpha */
        , mat1.m_data.data()  /* const double *a */
        , mat1.m_dim[1] /* const MKL_INT lda */
        , mat2.m_data.data()  /* const double *b */
        , mat2.m_dim[1] /* const MKL_INT ldb */
        , 0.0 /* const double beta */
        , ret.m_data.data() /* double * c */
        , ret.m_dim[1] /* const MKL_INT ldc */
        );
    return ret;
}

PYBIND11_MODULE(_array, m) {
    py::class_<Array<int>>(m, "Array")
        .def(py::init<vector<int>>())
        .def("__getitem__", [](Array<int> &self, size_t idx) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            return self.m_data[idx];
        })
        .def("__getitem__", [](Array<int> &self, vector<size_t> idx) {
            if (self.m_dim.size() != idx.size())
                throw std::range_error("the dimension of query indexes should be as same as the dimension of the array");
            size_t real_idx = 0;
            for (int i = idx.size()-1; i >= 0; i--)
            {
                real_idx *= self.m_dim[i];
                real_idx += idx[i];
            }
            return self.m_data[real_idx];
        })
        .def("__setitem__", [](Array<int> &self, size_t idx, const int value) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            self.m_data[idx] = value;
        })
        .def("__setitem__", [](Array<int> &self, vector<size_t> idx, const int value) {
            if (self.m_dim.size() != idx.size())
                throw std::range_error("the dimension of query indexes should be as same as the dimension of the array");
            size_t real_idx = 0;
            for (int i = idx.size()-1; i >= 0; i--)
            {
                real_idx *= self.m_dim[i];
                real_idx += idx[i];
            }
            self.m_data[real_idx] = value;
        })
        .def("reshape", &Array<int>::reshape)
        .def("__add__", [](Array<int> &a1, Array<int> &a2){ return a1 + a2;})
        .def("__sub__", [](Array<int> &a1, Array<int> &a2){ return a1 - a2;})
        .def("__mul__", [](Array<int> &a1, Array<int> &a2){ return a1 * a2;})
        // .def("__matmul__", &matmul<int>)
        .def("__eq__", [](Array<int> &a1, Array<int> &a2){ return a1 == a2;})
        .def("__ne__", [](Array<int> &a1, Array<int> &a2){ return !(a1 == a2);})
        .def_readonly("size", &Array<int>::m_size)
        .def_readonly("dim", &Array<int>::m_dim);
    m.def("zeros", &zeros<int>);
    m.def("ones", &ones<int>);

    py::class_<Array<double>>(m, "ArrayF")
        .def(py::init<vector<double>>())
        .def("__getitem__", [](Array<double> &self, size_t idx) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            return self.m_data[idx];
        })
        .def("__getitem__", [](Array<double> &self, vector<size_t> idx) {
            if (self.m_dim.size() != idx.size())
                throw std::range_error("the dimension of query indexes should be as same as the dimension of the array");
            size_t real_idx = 0;
            for (int i = idx.size()-1; i >= 0; i--)
            {
                real_idx *= self.m_dim[i];
                real_idx += idx[i];
            }
            return self.m_data[real_idx];
        })
        .def("__setitem__", [](Array<double> &self, size_t idx, const double value) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            self.m_data[idx] = value;
        })
        .def("__setitem__", [](Array<double> &self, vector<size_t> idx, const double value) {
            if (self.m_dim.size() != idx.size())
                throw std::range_error("the dimension of query indexes should be as same as the dimension of the array");
            size_t real_idx = 0;
            for (int i = idx.size()-1; i >= 0; i--)
            {
                real_idx *= self.m_dim[i];
                real_idx += idx[i];
            }
            self.m_data[real_idx] = value;
        })
        .def("reshape", &Array<double>::reshape)
        .def("__add__", [](Array<double> &a1, Array<double> &a2){ return a1 + a2;})
        .def("__sub__", [](Array<double> &a1, Array<double> &a2){ return a1 - a2;})
        .def("__mul__", [](Array<double> &a1, Array<double> &a2){ return a1 * a2;})
        .def("__matmul__", &matmul)
        .def("__eq__", [](Array<double> &a1, Array<double> &a2){ return a1 == a2;})
        .def("__ne__", [](Array<double> &a1, Array<double> &a2){ return !(a1 == a2);})
        .def_readonly("size", &Array<double>::m_size)
        .def_readonly("dim", &Array<double>::m_dim);
    m.def("zerosF", &zeros<double>);
    m.def("onesF", &ones<double>);

}