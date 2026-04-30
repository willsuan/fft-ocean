#include "Ocean.h"

#include "kiss_fftnd.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace {
// box-muller -> two independent N(0,1) samples packed as a complex
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

void Ocean::setLayerWeight(size_t i, float w) {
    if (i < p_.layers.size())  p_.layers[i].weight   = w;
    if (i < patches_.size())   patches_[i].layer.weight = w;
}

void Ocean::reseed(const Params& p) {
    p_ = p;
    if (p_.tile < 1) p_.tile = 1;

    // default cascade: swell + chop + ripple
    // Ls are big primes so they don't line up -> no obvious tiling
    // wind directions are offset so crests don't all run the same way
    if (p_.layers.empty()) {
        Layer swell;  swell.L  = 1297.f; swell.windSpeed = 20.f; swell.amplitude  = 0.6f;  swell.weight  = 1.0f; swell.windDirDeg  =   0.f; swell.seed  = 1337;
        Layer chop;   chop.L   =  599.f; chop.windSpeed  =  8.f; chop.amplitude   = 0.4f;  chop.weight   = 1.0f; chop.windDirDeg   =  25.f; chop.seed   = 2024;
        Layer ripple; ripple.L =  127.f; ripple.windSpeed=  3.f; ripple.amplitude = 0.15f; ripple.weight = 1.0f; ripple.windDirDeg = -30.f; ripple.seed = 31415;
        p_.layers = { swell, chop, ripple };
    }
    buildMesh_();
    allocBuffersAndSpectrum_();
}

void Ocean::buildMesh_() {
    const int   N   = p_.N;
    const int   K   = p_.tile;
    const float Ts  = p_.tileSize;

    const int   side  = K * N + 1;
    const float total = K * Ts;
    const float dx    = Ts / static_cast<float>(N);
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
    ifft_ = kiss_fftnd_alloc(dims, 2, 1, nullptr, nullptr);

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
        patches_[i].h_real.assign(NN, 0.0f);
        patches_[i].dx_real.assign(NN, 0.0f);
        patches_[i].dz_real.assign(NN, 0.0f);
        initLayer_(patches_[i]);
    }
}

// philips spectrum P(k). returns 0 at k=0
float Ocean::philips_(const Layer& layer, float kx, float kz) const {
    const float k2 = kx * kx + kz * kz;
    if (k2 < 1e-12f) return 0.0f;
    const float V  = layer.windSpeed;
    const float g  = p_.gravity;
    const float Lw = V * V / g;                  // largest wave the wind can drive
    const float k  = std::sqrt(k2);
    const float wx = std::cos(layer.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float wz = std::sin(layer.windDirDeg * static_cast<float>(M_PI) / 180.0f);
    const float kdotw = (kx * wx + kz * wz) / k;
    float P = layer.amplitude * std::exp(-1.0f / (k2 * Lw * Lw)) / (k2 * k2)
              * (kdotw * kdotw);
    // damp upwind waves a bit (Tessendorf eqn 24)
    if (kdotw < 0.0f) P *= 0.07f;
    // damp very high-k waves to keep the surface from buzzing
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

    // h0(k) = (1/sqrt 2) * (xi_r + i xi_i) * sqrt(P(k))
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
    // pre-compute conj(h0(-k)) so update() doesn't need to look it up
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int mi = (N - i) % N;
            int mj = (N - j) % N;
            s.h0_conj[i * N + j] = std::conj(s.h0[mi * N + mj]);
        }
    }
}

void Ocean::update(float t, float choppiness, float foamThreshold,
                   float displayGain) {
    const int N    = p_.N;
    const int K    = p_.tile;
    const int side = K * N + 1;

    const std::complex<float> minus_i(0.0f, -1.0f);
    auto modeIdx = [&](int i) { return i - N / 2; };

    // for each layer: build h(k,t) and the chop displacement specs, run IFFTs
    for (auto& s : patches_) {
        const float twoPiOverL = 2.0f * static_cast<float>(M_PI) / s.layer.L;
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const int idx = i * N + j;
                // h(k,t) = h0 e^{i w t} + conj(h0(-k)) e^{-i w t}
                float w  = s.omega[idx] * t;
                float c  = std::cos(w);
                float sn = std::sin(w);
                std::complex<float> e_pos{ c,  sn};
                std::complex<float> e_neg{ c, -sn};
                std::complex<float> h = s.h0[idx] * e_pos + s.h0_conj[idx] * e_neg;
                s.ht[idx] = h;

                // chop: D = IFFT[-i k_hat h]
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

        // (-1)^(i+j) sign flip undoes the centered-k convention,
        // 1/N^2 normalizes the unnormalized inverse FFT
        const float invNN = 1.0f / static_cast<float>(N * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                const int idx = i * N + j;
                const float sgn = ((i + j) & 1) ? -1.0f : 1.0f;
                s.h_real[idx]  = sgn * s.h_out[idx].real()  * invNN;
                s.dx_real[idx] = sgn * s.dx_out[idx].real() * invNN;
                s.dz_real[idx] = sgn * s.dz_out[idx].real() * invNN;
            }
        }
    }

    const float Ts        = p_.tileSize;
    const float dxPhys    = Ts / static_cast<float>(N);
    const float totalHalf = 0.5f * static_cast<float>(K) * Ts;

    // bilinear sample of a periodic field, u,v in [0,1)
    // (using nearest-neighbor here gives obvious blocky tile artifacts)
    auto bilerp = [&](const std::vector<float>& field, float u, float v) -> float {
        float fi = u * N;
        float fj = v * N;
        int   i0 = static_cast<int>(std::floor(fi));
        int   j0 = static_cast<int>(std::floor(fj));
        float ti = fi - i0;
        float tj = fj - j0;
        int   ia = ((i0    ) % N + N) % N;
        int   ib = ((i0 + 1) % N + N) % N;
        int   ja = ((j0    ) % N + N) % N;
        int   jb = ((j0 + 1) % N + N) % N;
        float f00 = field[ia * N + ja];
        float f10 = field[ib * N + ja];
        float f01 = field[ia * N + jb];
        float f11 = field[ib * N + jb];
        float f0  = (1.0f - tj) * f00 + tj * f01;
        float f1  = (1.0f - tj) * f10 + tj * f11;
        return (1.0f - ti) * f0 + ti * f1;
    };

    dxTotal_.assign(side * side, 0.0f);
    dzTotal_.assign(side * side, 0.0f);

    // build the actual mesh: sum the layers at each world position
    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            float world_x = static_cast<float>(I) * dxPhys - totalHalf;
            float world_z = static_cast<float>(J) * dxPhys - totalHalf;

            float hSum = 0.0f, dxSum = 0.0f, dzSum = 0.0f;
            for (const auto& s : patches_) {
                const float Llay = s.layer.L;
                const float w    = s.layer.weight;
                float u = world_x / Llay;
                float v = world_z / Llay;
                u -= std::floor(u);   // wrap to [0,1)
                v -= std::floor(v);
                hSum  += w * bilerp(s.h_real,  u, v);
                dxSum += w * bilerp(s.dx_real, u, v);
                dzSum += w * bilerp(s.dz_real, u, v);
            }

            int dst = I * side + J;
            // bake displayGain into both the height and the chop displacement
            // so a strong vgain doesn't visually flatten the chop sharpening
            float Dx = displayGain * choppiness * dxSum;
            float Dz = displayGain * choppiness * dzSum;
            dxTotal_[dst] = Dx;
            dzTotal_[dst] = Dz;

            V_(dst, 0) = world_x - Dx;
            V_(dst, 1) = displayGain * hSum;
            V_(dst, 2) = world_z - Dz;
        }
    }

    // jacobian foam: J = (1+dDx/dx)(1+dDz/dz) - (dDx/dz)(dDz/dx)
    // when J<0 the displacement field is folding -> wave is breaking
    auto idx = [&](int I, int J) { return I * side + J; };
    auto dDxdx = [&](int I, int J) -> float {
        if (I == 0)        return (dxTotal_[idx(1, J)]      - dxTotal_[idx(0, J)])     / dxPhys;
        if (I == side - 1) return (dxTotal_[idx(side-1, J)] - dxTotal_[idx(side-2, J)]) / dxPhys;
        return (dxTotal_[idx(I+1, J)] - dxTotal_[idx(I-1, J)]) / (2.0f * dxPhys);
    };
    auto dDxdz = [&](int I, int J) -> float {
        if (J == 0)        return (dxTotal_[idx(I, 1)]      - dxTotal_[idx(I, 0)])     / dxPhys;
        if (J == side - 1) return (dxTotal_[idx(I, side-1)] - dxTotal_[idx(I, side-2)]) / dxPhys;
        return (dxTotal_[idx(I, J+1)] - dxTotal_[idx(I, J-1)]) / (2.0f * dxPhys);
    };
    auto dDzdx = [&](int I, int J) -> float {
        if (I == 0)        return (dzTotal_[idx(1, J)]      - dzTotal_[idx(0, J)])     / dxPhys;
        if (I == side - 1) return (dzTotal_[idx(side-1, J)] - dzTotal_[idx(side-2, J)]) / dxPhys;
        return (dzTotal_[idx(I+1, J)] - dzTotal_[idx(I-1, J)]) / (2.0f * dxPhys);
    };
    auto dDzdz = [&](int I, int J) -> float {
        if (J == 0)        return (dzTotal_[idx(I, 1)]      - dzTotal_[idx(I, 0)])     / dxPhys;
        if (J == side - 1) return (dzTotal_[idx(I, side-1)] - dzTotal_[idx(I, side-2)]) / dxPhys;
        return (dzTotal_[idx(I, J+1)] - dzTotal_[idx(I, J-1)]) / (2.0f * dxPhys);
    };

    // foamThreshold is the J value below which we paint full foam.
    // J is ~1 on flat water and shrinks (possibly going negative) where the
    // displacement field compresses or folds. crossfade is 0.3 wide.
    const float cutHigh = foamThreshold;
    const float cutLow  = cutHigh - 0.3f;
    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            float Jxx = dDxdx(I, J);
            float Jzz = dDzdz(I, J);
            float Jxz = dDxdz(I, J);
            float Jzx = dDzdx(I, J);
            float Jdet = (1.0f + Jxx) * (1.0f + Jzz) - Jxz * Jzx;
            float f = (cutHigh - Jdet) / (cutHigh - cutLow);
            foam_(idx(I, J)) = std::max(0.0f, std::min(1.0f, f));
        }
    }

    // single 3x3 box-blur pass: enough to take out the per-vertex stepping
    // without smearing foam across whole regions.
    std::vector<float> blurred(side * side, 0.0f);
    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            float sum = 0.0f;
            int   n   = 0;
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    int ii = I + di, jj = J + dj;
                    if (ii < 0 || ii >= side || jj < 0 || jj >= side) continue;
                    sum += static_cast<float>(foam_(idx(ii, jj)));
                    ++n;
                }
            }
            blurred[idx(I, J)] = sum / std::max(1, n);
        }
    }
    for (int v = 0; v < side * side; ++v) foam_(v) = blurred[v];
}
