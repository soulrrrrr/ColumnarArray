#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>

using std::vector;

namespace py = pybind11;

struct Array {
    Array(){}
    Array(vector<int> v): m_data(v), m_size(v.size()) {}

    size_t m_size;
    vector<int> m_data;
};

PYBIND11_MODULE(example, m) {
    py::class_<Array>(m, "_array")
        .def(py::init<vector<int>>())
        .def_readonly("size", &Array::m_size)
}