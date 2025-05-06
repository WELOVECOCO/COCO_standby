#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <complex>

namespace py = pybind11;
using Complex = std::complex<double>;

class FFTConvolver {
public:
    /// input: 4D array (N, C, H, W)
    /// kernel: 3D array (C, Kh, Kw)
    /// returns: 4D array (N, 1, H-Kh+1, W-Kw+1)
    py::array_t<double> convolve(py::array_t<double> input,
                                 py::array_t<double> kernel);

private:
    void fft1d(std::vector<Complex>& a, bool invert);
    void fft2d(std::vector<std::vector<Complex>>& m, bool invert);

    static int nextPow2(int x) {
        int p = 1;
        while (p < x) p <<= 1;
        return p;
    }
};
