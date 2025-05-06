#define _USE_MATH_DEFINES
#include "fft_convolver.h"
#include <cmath>
#include <algorithm>

void FFTConvolver::fft1d(std::vector<Complex>& a, bool invert) {
    int n = (int)a.size();
    if (n < 2) return;
    // bit-reverse
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    // Cooley–Tukey
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * M_PI / len * (invert ? -1 : 1);
        Complex wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int k = 0; k < len/2; ++k) {
                Complex u = a[i+k];
                Complex v = a[i+k+len/2] * w;
                a[i+k] = u + v;
                a[i+k+len/2] = u - v;
                w *= wlen;
            }
        }
    }
    if (invert) {
        for (auto & x : a) x /= n;
    }
}

void FFTConvolver::fft2d(std::vector<std::vector<Complex>>& m, bool invert) {
    int H = (int)m.size(), W = (int)m[0].size();
    // rows
    for (int i = 0; i < H; ++i) fft1d(m[i], invert);
    // cols
    std::vector<Complex> col(H);
    for (int j = 0; j < W; ++j) {
        for (int i = 0; i < H; ++i) col[i] = m[i][j];
        fft1d(col, invert);
        for (int i = 0; i < H; ++i) m[i][j] = col[i];
    }
}

py::array_t<double> FFTConvolver::convolve(py::array_t<double> input,
                                           py::array_t<double> kernel) {
    auto in_buf = input.request();
    auto kr_buf = kernel.request();
    if (in_buf.ndim != 4 || kr_buf.ndim != 3)
        throw std::runtime_error("Expect input(N,C,H,W) & kernel(C,Kh,Kw)");

    int N  = (int)in_buf.shape[0];
    int C  = (int)in_buf.shape[1];
    int H  = (int)in_buf.shape[2];
    int W  = (int)in_buf.shape[3];
    int Kh = (int)kr_buf.shape[1];
    int Kw = (int)kr_buf.shape[2];

    int outH = H - Kh + 1;
    int outW = W - Kw + 1;
    if (outH <= 0 || outW <= 0)
        throw std::runtime_error("Kernel larger than input.");

    // FFT dims
    int Hfft = nextPow2(H + Kh - 1);
    int Wfft = nextPow2(W + Kw - 1);

    // Prepare result buffer: shape (N,1,outH,outW)
    py::array_t<double> result({N, 1, outH, outW});
    auto res_buf = result.mutable_unchecked<4>();

    // Temp per-channel sum
    std::vector<std::vector<double>> sum2d(outH, std::vector<double>(outW));

    // Pointers to raw data
    double* in_ptr = (double*)in_buf.ptr;
    double* kr_ptr = (double*)kr_buf.ptr;

    for (int n = 0; n < N; ++n) {
        // zero sum
        for (int i = 0; i < outH; ++i)
            for (int j = 0; j < outW; ++j)
                sum2d[i][j] = 0.0;

        for (int c = 0; c < C; ++c) {
            // Build complex mats A, B
            std::vector<std::vector<Complex>> A(Hfft, std::vector<Complex>(Wfft, 0));
            std::vector<std::vector<Complex>> B(Hfft, std::vector<Complex>(Wfft, 0));

            // Copy input[c]
            int offset_in = ((n*C + c)*H)*W;
            for (int i = 0; i < H; ++i)
                for (int j = 0; j < W; ++j)
                    A[i][j] = in_ptr[offset_in + i*W + j];

            // Copy kernel[c] (no flip: cross-corr)
            int offset_kr = (c*Kh)*Kw;
            for (int i = 0; i < Kh; ++i)
                for (int j = 0; j < Kw; ++j)
                    B[i][j] = kr_ptr[offset_kr + i*Kw + j];

            // FFT, multiply, IFFT
            fft2d(A, false);
            fft2d(B, false);
            for (int i = 0; i < Hfft; ++i)
                for (int j = 0; j < Wfft; ++j)
                    A[i][j] *= B[i][j];
            fft2d(A, true);

            // extract valid region and accumulate
            for (int i = 0; i < outH; ++i)
                for (int j = 0; j < outW; ++j)
                    sum2d[i][j] += A[i + Kh - 1][j + Kw - 1].real();
        }

        // write to result
        for (int i = 0; i < outH; ++i)
            for (int j = 0; j < outW; ++j)
                res_buf(n, 0, i, j) = sum2d[i][j];
    }

    return result;
}
