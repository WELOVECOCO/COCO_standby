#include <pybind11/pybind11.h>
#include "fft_convolver.h"
namespace py = pybind11;

PYBIND11_MODULE(FasterCoco, m) {
    py::class_<FFTConvolver>(m, "FFTConvolver")
        .def(py::init<>())
        .def("convolve", &FFTConvolver::convolve,
             py::arg("input"), py::arg("kernel"),
             "Perform valid cross-correlation on (N,C,H,W) input with (C,Kh,Kw) kernels");
}
