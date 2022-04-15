#include <pybind11/pybind11.h>
namespace py = pybind11;

/* Ignore. The python to_bytes() method is faster */
py::bytes int_to_bytes(long *x){
    return py::bytes(static_cast<char*>(static_cast<void*>(x)));
}

PYBIND11_MODULE(hasher, m) {
    m.def("int_to_bytes", &int_to_bytes, py::return_value_policy::reference);
}