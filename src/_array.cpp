#include <iostream>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
using std::vector;

namespace py = pybind11;

void check_dim(vector<size_t> v1, vector<size_t> v2)
{
    if (v1 != v2)
        throw std::range_error("the dimension of 2 array should be same");
}


class Array {
public:
    Array(): m_size(0) {}
    Array(vector<int> v): m_data(v), m_size(v.size()), m_dim(vector<size_t>(1, v.size())) {}
    Array(vector<int> v, vector<size_t> dim): m_data(v), m_size(v.size()), m_dim(dim) {}

    Array operator+(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < m_size; i++)
            ret.m_data[i] += v.m_data[i];
        return ret;
    }

    Array operator-(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < m_size; i++)
            ret.m_data[i] -= v.m_data[i];
        return ret;
    }

    Array operator*(Array v)
    {
        check_dim(m_dim, v.m_dim);
        Array ret(m_data, m_dim);
        for (int i = 0; i < m_size; i++)
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

    Array arange(const vector<size_t>& v)
    {
        size_t size = 1;
        for (auto i : v)
            size *= i;
        if (m_size != size)
            throw std::range_error("the size of 2 arrays are not same");

        m_dim.clear();
        for (auto i : v)
            m_dim.push_back(i);

        return *this;
    }

public:
    size_t m_size;
    vector<size_t> m_dim;
    vector<int> m_data;
};

Array zeros(vector<size_t> dim)
{
    size_t size = 1;
    for (size_t d : dim)
        size *= d;
    return Array(vector<int>(size, 0), dim);
}

Array ones(vector<size_t> dim)
{
    size_t size = 1;
    for (size_t d : dim)
        size *= d;
    return Array(vector<int>(size, 1), dim);
}

PYBIND11_MODULE(_array, m) {
    py::class_<Array>(m, "Array")
        .def(py::init<vector<int>>())
        .def("__getitem__", [](Array &self, size_t idx) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            return self.m_data[idx];
        })
        .def("__getitem__", [](Array &self, vector<size_t> idx) {
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
        .def("__setitem__", [](Array &self, size_t idx, const int value) {
            if (self.m_size <= idx)
                throw std::range_error("index too large");
            self.m_data[idx] = value;
        })
        .def("__setitem__", [](Array &self, vector<size_t> idx, const int value) {
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
        .def("show", [](Array &self){
            for (auto i : self.m_data)
                py::print(i);
        })
        .def("arange", &Array::arange)
        .def("__add__", [](Array &a1, Array &a2){ return a1 + a2;})
        .def("__sub__", [](Array &a1, Array &a2){ return a1 - a2;})
        .def("__mul__", [](Array &a1, Array &a2){ return a1 * a2;})
        .def("__eq__", [](Array &a1, Array &a2){ return a1 == a2;})
        .def_readonly("size", &Array::m_size);
    m.def("zeros", &zeros);
    m.def("ones", &ones);

}