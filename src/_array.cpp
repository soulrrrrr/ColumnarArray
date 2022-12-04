#include <iostream>
#include <vector>
#include <utility>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using std::vector;

namespace py = pybind11;

class Array {
public:
    Array(){}
    Array(vector<int> v): m_data(v), m_size(v.size()), m_dim(vector<size_t>(1, v.size())) {}
public:
    size_t m_size;
    vector<size_t> m_dim;
    vector<int> m_data;
};

PYBIND11_MODULE(_array, m) {
    py::class_<Array>(m, "Array")
        .def(py::init<vector<int>>())
        .def("__getitem__", [](Array &self, size_t idx) {
            return self.m_data[idx];
        })
        .def("__getitem__", [](Array &self, vector<size_t> idx) {
            return self.m_data[idx[0]];
        })
        .def("__setitem__", [](Array &self, size_t idx, const int value) {
            self.m_data[idx] = value;
        })
        .def("__setitem__", [](Array &self, vector<size_t> idx, const int value) {
            self.m_data[idx[0]] = value;
        })
        .def("show", [](Array &self){
            for (auto i : self.m_data)
                py::print(i);
        })
        .def_readonly("size", &Array::m_size);

}