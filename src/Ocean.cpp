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
    if (p_.layers.empty()) {
        // Default 3-layer cascade: long swell / mid chop / short ripples.
        Layer swell;  swell.L  = 250.f; swell.amplitude  = 2.0f; swell.weight = 1.0f; swell.seed = 1337;
        Layer chop;   chop.L   = 60.f;  chop.amplitude   = 0.6f; chop.weight  = 1.0f; chop.seed  = 2024;
        Layer ripple; ripple.L = 15.f;  ripple.amplitude = 0.15f; ripple.weight = 1.0f; ripple.seed = 31415;
        p_.layers = { swell, chop, ripple };
    }
    buildMesh_();
    allocBuffersAndSpectrum_();
}

void Ocean::buildMesh_() {
    const int   N = p_.N;
    const int   K = p_.tile;

    // Use the LARGEST patch L as the world-tile size so all cascade layers
    // fit seamlessly into the rendered area.
    float Lmax = 1.0f;
    for (const auto& layer : p_.layers) Lmax = std::max(Lmax, layer.L);

    const int   side  = K * N + 1;
    const float total = K * Lmax;
    const float dx    = Lmax / static_cast<float>(N);
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

void Ocean::allocBuffersAndSpectrum_() {
    if (ifft_) { kiss_fft_free(ifft_); ifft_ = nullptr; }
    int dims[2] = { p_.N, p_.N };
    ifft_ = kiss_fftnd_alloc(dims, 2, /*inverse=*/1, nullptr, nullptr);

    patches_.clear();
    patches_.resize(p_.layers.size());
    const int NN = p_.N * p_.N;
    for (size_t i = 0; i < p_.layers.size(); ++i) {
        patches_[i].layer = p_.layers[i];
        patches_[i].h0.assign(NN, {0, 0});
        patches_[i].h0_conj.assign(NN, {0, 0});
        patches_[i].omega.assign(NN, 0.0f);
        patches_[i].ht.assign(NN, {0, 0});
        patches_[i].dx_spec.assign(NN, {0, 0});
        patches_[i].dz_spec.assign(NN, {0, 0});
        patches_[i].h_out.assign(NN, {0, 0});
        patches_[i].dx_out.assign(NN, {0, 0});
        patches_[i].dz_out.assign(NN, {0, 0});
        initLayer_(patches_[i]);
    }
}

float Ocean::philips_(const Layer& layer, float kx, float kz) const {
    const float k2 = kx * kx + kz * kz;
    if (k2 < 1e-12f) return 0.0f;
    const float V  = layer.windSpeed;
    const float g  = p_.gravity;
    const float Lw = V * V / g;
    const float k  = std::sqrt(k2);
    const float wx = std::cos(p_.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float wz = std::sin(p_.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float kdotw = (kx * wx + kz * wz) / k;
    float P = layer.amplitude * std::exp(-1.0f / (k2 * Lw * Lw)) / (k2 * k2)
              * (kdotw * kdotw);
    if (kdotw < 0.0f) P *= 0.07f;
    const float l = Lw * p_.cutoff;
    P *= std::exp(-k2 * l * l);
    return P;
}

void Ocean::initLayer_(PatchState& s) {
    const int   N  = p_.N;
    const float L  = s.layer.L;
    const float twoPiOverL = 2.0f * static_cast<float>(M_PI) / L;

    std::mt19937 rng(s.layer.seed);
    auto modeIdx = [&](int i) { return i - N / 2; };

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float kx = modeIdx(i) * twoPiOverL;
            float kz = modeIdx(j) * twoPiOverL;
            float P  = philips_(s.layer, kx, kz);
            std::complex<float> g = gaussianComplex(rng);
            s.h0[i * N + j]    = g * std::sqrt(P / 2.0f);
            s.omega[i * N + j] = std::sqrt(p_.gravity * std::sqrt(kx*kx + kz*kz));
        }
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int mi = (N - i) % N;
            int mj = (N - j) % N;
            s.h0_conj[i * N + j] = std::conj(s.h0[mi * N + mj]);
        }
    }
}

void Ocean::update(float t, float choppiness, float foamThreshold) {
    const int N    = p_.N;
    const int K    = p_.tile;
    const int side = K * N + 1;

    // Evolve each cascade patch independently.
    const std::complex<float> minus_i(0.0f, -1.0f);
    auto modeIdx = [&](int i) { return i - N / 2; };
    for (auto& s : patches_) {
        const float twoPiOverL = 2.0f * static_cast<float>(M_PI) / s.layer.L;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const int idx = i * N + j;
                float w  = s.omega[idx] * t;
                float c  = std::cos(w);
                float sn = std::sin(w);
                std::complex<float> e_pos{ c,  sn};
                std::complex<float> e_neg{ c, -sn};
                std::complex<float> h = s.h0[idx] * e_pos + s.h0_conj[idx] * e_neg;
                s.ht[idx] = h;
                float kx = modeIdx(i) * twoPiOverL;
                float kz = modeIdx(j) * twoPiOverL;
                float k  = std::sqrt(kx*kx + kz*kz);
                if (k < 1e-6f) {
                    s.dx_spec[idx] = {0, 0};
                    s.dz_spec[idx] = {0, 0};
                } else {
                    s.dx_spec[idx] = minus_i * (kx / k) * h;
                    s.dz_spec[idx] = minus_i * (kz / k) * h;
                }
            }
        }
        kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(s.ht.data()),
                          reinterpret_cast<kiss_fft_cpx*>(s.h_out.data()));
        kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(s.dx_spec.data()),
                          reinterpret_cast<kiss_fft_cpx*>(s.dx_out.data()));
        kiss_fftnd(ifft_, reinterpret_cast<const kiss_fft_cpx*>(s.dz_spec.data()),
                          reinterpret_cast<kiss_fft_cpx*>(s.dz_out.data()));
    }

    // Largest patch defines the world tile size.
    float Lmax = 1.0f;
    for (const auto& layer : p_.layers) Lmax = std::max(Lmax, layer.L);
    const float dxPhys    = Lmax / static_cast<float>(N);
    const float totalHalf = 0.5f * static_cast<float>(K) * Lmax;
    const float invNN     = 1.0f / static_cast<float>(N * N);

    // Build composite surface: for each mesh vertex, query each cascade layer
    // at the same world position and sum the contributions. Each layer is
    // periodic at its own L, so we sample via (world_x / L_layer) mod N —
    // different layers have different periods and the sum appears non-periodic.
    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            float world_x = static_cast<float>(I) * dxPhys - totalHalf;
            float world_z = static_cast<float>(J) * dxPhys - totalHalf;

            float hSum = 0.0f, dxSum = 0.0f, dzSum = 0.0f;
            for (const auto& s : patches_) {
                const float Llay = s.layer.L;
                const float w    = s.layer.weight;
                // Map world (x,z) into this layer's N×N periodic grid.
                float u = world_x / Llay;
                float v = world_z / Llay;
                u -= std::floor(u);
                v -= std::floor(v);
                int i = static_cast<int>(u * N) % N;
                int j = static_cast<int>(v * N) % N;
                if (i < 0) i += N;
                if (j < 0) j += N;
                int idx = i * N + j;
                float sign = ((i + j) & 1) ? -1.0f : 1.0f;
                hSum  += w * sign * s.h_out[idx].real()  * invNN;
                dxSum += w * sign * s.dx_out[idx].real() * invNN;
                dzSum += w * sign * s.dz_out[idx].real() * invNN;
            }

            int dst = I * side + J;
            V_(dst, 0) = world_x - choppiness * dxSum;
            V_(dst, 1) = hSum;
            V_(dst, 2) = world_z - choppiness * dzSum;
        }
    }

    // Height-based foam over the tiled composite.
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
