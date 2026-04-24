#include "Ocean.h"

#include "kiss_fftnd.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

namespace {
inline std::complex<float> gaussianComplex(std::mt19937& rng) {
    std::uniform_real_distribution<float> U(1e-7f, 1.0f);
    float u1 = U(rng), u2 = U(rng);
    float r  = std::sqrt(-2.0f * std::log(u1));
    float th = 2.0f * static_cast<float>(M_PI) * u2;
    return { r * std::cos(th), r * std::sin(th) };
}
} // namespace

Ocean::Ocean(const Params& p) { reseed(p); }

Ocean::~Ocean() {
    if (ifft_) { kiss_fft_free(ifft_); ifft_ = nullptr; }
}

void Ocean::reseed(const Params& p) {
    p_ = p;
    if (p_.tile < 1) p_.tile = 1;
    buildMesh_();
    allocBuffers_();
    computeInitialSpectrum_();
}

void Ocean::buildMesh_() {
    const int   N = p_.N;
    const float L = p_.L;
    const int   K = p_.tile;

    // Mesh spans K*L world units in X and Z, sampled on a (K*N+1)² grid. We
    // use K*N+1 (not K*N) so the seam row/column is present — its positions
    // will match the opposite edge of the periodic heightfield, making tile
    // joins invisible.
    const int   side  = K * N + 1;
    const float total = K * L;
    const float dx    = L / static_cast<float>(N); // periodic spacing
    const float half  = 0.5f * total;

    V_.resize(side * side, 3);
    F_.resize(2 * (side - 1) * (side - 1), 3);
    foam_ = Eigen::VectorXd::Zero(side * side);

    for (int i = 0; i < side; ++i) {
        for (int j = 0; j < side; ++j) {
            V_(i * side + j, 0) = static_cast<float>(i) * dx - half;
            V_(i * side + j, 1) = 0.0f;
            V_(i * side + j, 2) = static_cast<float>(j) * dx - half;
        }
    }
    int f = 0;
    for (int i = 0; i < side - 1; ++i) {
        for (int j = 0; j < side - 1; ++j) {
            int v00 = i       * side + j;
            int v10 = (i + 1) * side + j;
            int v01 = i       * side + (j + 1);
            int v11 = (i + 1) * side + (j + 1);
            F_.row(f++) << v00, v10, v11;
            F_.row(f++) << v00, v11, v01;
        }
    }
}

void Ocean::allocBuffers_() {
    if (ifft_) { kiss_fft_free(ifft_); ifft_ = nullptr; }
    int dims[2] = { p_.N, p_.N };
    ifft_ = kiss_fftnd_alloc(dims, 2, /*inverse=*/1, nullptr, nullptr);
    const int NN = p_.N * p_.N;
    h0_.assign(NN, {0, 0});
    h0_conj_.assign(NN, {0, 0});
    omega_.assign(NN, 0.0f);
    ht_.assign(NN, {0, 0});
    dx_spec_.assign(NN, {0, 0});
    dz_spec_.assign(NN, {0, 0});
    h_out_.assign(NN, {0, 0});
    dx_out_.assign(NN, {0, 0});
    dz_out_.assign(NN, {0, 0});
}

float Ocean::philips_(float kx, float kz) const {
    const float k2 = kx * kx + kz * kz;
    if (k2 < 1e-12f) return 0.0f;
    const float V  = p_.windSpeed;
    const float g  = p_.gravity;
    const float Lw = V * V / g;
    const float k  = std::sqrt(k2);
    const float wx = std::cos(p_.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float wz = std::sin(p_.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float kdotw = (kx * wx + kz * wz) / k;
    float P = p_.amplitude * std::exp(-1.0f / (k2 * Lw * Lw)) / (k2 * k2)
              * (kdotw * kdotw);
    if (kdotw < 0.0f) P *= 0.07f;
    const float l = Lw * p_.cutoff;
    P *= std::exp(-k2 * l * l);
    return P;
}

void Ocean::computeInitialSpectrum_() {
    const int   N = p_.N;
    const float L = p_.L;
    const float twoPiOverL = 2.0f * static_cast<float>(M_PI) / L;

    std::mt19937 rng(p_.seed);
    auto modeIdx = [&](int i) { return i - N / 2; };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float kx = modeIdx(i) * twoPiOverL;
            float kz = modeIdx(j) * twoPiOverL;
            float P  = philips_(kx, kz);
            std::complex<float> g = gaussianComplex(rng);
            h0_[i * N + j]    = g * std::sqrt(P / 2.0f);
            omega_[i * N + j] = std::sqrt(p_.gravity * std::sqrt(kx*kx + kz*kz));
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int mi = (N - i) % N;
            int mj = (N - j) % N;
            h0_conj_[i * N + j] = std::conj(h0_[mi * N + mj]);
        }
    }
}

void Ocean::update(float t, float choppiness, float foamThreshold) {
    const int   N = p_.N;
    const float L = p_.L;
    const int   K = p_.tile;
    const int   side = K * N + 1;
    const float twoPiOverL = 2.0f * static_cast<float>(M_PI) / L;

    const std::complex<float> minus_i(0.0f, -1.0f);
    auto modeIdx = [&](int i) { return i - N / 2; };
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            const int idx = i * N + j;
            float w  = omega_[idx] * t;
            float c  = std::cos(w);
            float s  = std::sin(w);
            std::complex<float> e_pos{ c,  s};
            std::complex<float> e_neg{ c, -s};
            std::complex<float> h = h0_[idx] * e_pos + h0_conj_[idx] * e_neg;
            ht_[idx] = h;
            float kx = modeIdx(i) * twoPiOverL;
            float kz = modeIdx(j) * twoPiOverL;
            float k  = std::sqrt(kx*kx + kz*kz);
            if (k < 1e-6f) {
                dx_spec_[idx] = {0, 0};
                dz_spec_[idx] = {0, 0};
            } else {
                dx_spec_[idx] = minus_i * (kx / k) * h;
                dz_spec_[idx] = minus_i * (kz / k) * h;
            }
        }
    }

    kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(ht_.data()),
                      reinterpret_cast<kiss_fft_cpx*>(h_out_.data()));
    kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(dx_spec_.data()),
                      reinterpret_cast<kiss_fft_cpx*>(dx_out_.data()));
    kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(dz_spec_.data()),
                      reinterpret_cast<kiss_fft_cpx*>(dz_out_.data()));

    // Scatter the NxN periodic result onto the (K*N+1)² tiled mesh via modulo
    // indexing. Vertices at tile seams end up with the same value at both
    // sides of the seam, so the rendered surface is seamless.
    const float norm = 1.0f / static_cast<float>(N * N);
    const float dxPhys = L / static_cast<float>(N);
    const float totalHalf = 0.5f * static_cast<float>(K) * L;

    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            int i = I % N;
            int j = J % N;
            int src = i * N + j;
            float sign = ((i + j) & 1) ? -1.0f : 1.0f;
            float h  = sign * h_out_[src].real()  * norm;
            float dx = sign * dx_out_[src].real() * norm * choppiness;
            float dz = sign * dz_out_[src].real() * norm * choppiness;
            int dst = I * side + J;
            V_(dst, 0) = static_cast<float>(I) * dxPhys - totalHalf - dx;
            V_(dst, 1) = h;
            V_(dst, 2) = static_cast<float>(J) * dxPhys - totalHalf - dz;
        }
    }

    // Height-based foam: normalize heights over the whole tiled surface and
    // mark as foam anything above the user's crest-percentile cutoff.
    double hMin =  std::numeric_limits<double>::infinity();
    double hMax = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < side * side; ++v) {
        double h = V_(v, 1);
        if (h < hMin) hMin = h;
        if (h > hMax) hMax = h;
    }
    const double hRange = std::max(hMax - hMin, 1e-6);
    const float edgeWidth = 0.08f;
    const float cutLow  = foamThreshold;
    const float cutHigh = std::min(1.0f, foamThreshold + edgeWidth);
    for (int v = 0; v < side * side; ++v) {
        double h_norm = (V_(v, 1) - hMin) / hRange;
        float f = static_cast<float>((h_norm - cutLow)
                                     / std::max(cutHigh - cutLow, 1e-6f));
        foam_(v) = std::max(0.0f, std::min(1.0f, f));
    }
}
