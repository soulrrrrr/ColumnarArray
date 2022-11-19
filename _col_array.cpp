#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace std;

template<typename T>
class Array {
public:
    Array(vector<T> data)
        :m_data(data)
    {
        m_exist.resize(data.size(), true);
        m_len = data.size();
    }

    int len() {return m_len;}
    vector<bool> m_exist;
    vector<T> m_data;
    int m_len;
};


// int main()
// {
//     cout << myMax<int>(3, 7) << endl;
//     cout << myMax<double>(3.3434, 7.999909099) << endl;
//     return 0;
// }

PYBIND11_MODULE(_col_array, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::class_<Array<int>>(m, "IntArray")
        .def(py::init<vector<int>>())
        .def("__getitem__", [](const Array<int> &arr, int idx){return arr.m_exist[idx] ? arr.m_data[idx] : -1;})
        .def_property_readonly("len", &Array<int>::len);

    py::class_<Array<double>>(m, "FloatArray")
        .def(py::init<vector<double>>())
        .def("__getitem__", [](const Array<double> &arr, int idx){return arr.m_exist[idx] ? arr.m_data[idx] : -1;})
        .def_property_readonly("len", &Array<double>::len);
}

