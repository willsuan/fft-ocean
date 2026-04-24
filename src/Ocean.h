// Ocean.h — Tessendorf FFT ocean heightfield simulation with seamless tiling.
//
// Based on Jerry Tessendorf, "Simulating Ocean Water" (SIGGRAPH 2001 course
// notes). The sea surface is modelled as a sum of Fourier modes whose
// amplitudes are drawn from the Philips wind-wave spectrum; time evolution is
// a per-mode phase rotation by ω(k)=√(g|k|); the heightfield at any instant
// is recovered by a 2D inverse FFT.
//
// Features:
//  - Philips wind-wave spectrum with directional preference and upwind damping
//  - Complex Gaussian initial amplitudes (Box–Muller)
//  - Time evolution preserving a real-valued surface
//  - Choppy (sharp-crested) waves via two extra IFFTs for horizontal displacement
//  - Height-based foam proxy for whitecaps
//  - K×K seamless tiling so a single N×N simulated patch renders as an
//    apparently-infinite ocean (FFT results are spatially periodic by
//    construction, so the tiles join with no visible seams).
#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>

struct kiss_fftnd_state;

class Ocean {
public:
    struct Params {
        int   N           = 128;      // grid resolution (power of 2)
        float L           = 250.0f;   // physical patch size (meters)
        int   tile        = 3;        // K: render K×K tiled copies of the patch
        float windSpeed   = 20.0f;
        float windDirDeg  = 0.0f;
        float amplitude   = 1.5f;
        float gravity     = 9.81f;
        float cutoff      = 0.001f;
        unsigned seed     = 1337;
    };

    Ocean() : Ocean(Params{}) {}
    explicit Ocean(const Params& p);
    ~Ocean();

    void reseed(const Params& p);
    void update(float t, float choppiness, float foamThreshold);

    const Eigen::MatrixXd& vertices() const { return V_; }
    const Eigen::MatrixXi& faces()    const { return F_; }
    const Eigen::VectorXd& foam()     const { return foam_; }

    const Params& params() const { return p_; }

private:
    Params p_;

    // Render mesh: (K*N + 1)² vertices so that adjacent tiles share edges and
    // the last row/column wraps to the first — no visible seams.
    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    Eigen::VectorXd foam_;

    // Spectrum state, size N*N (the simulated patch).
    std::vector<std::complex<float>> h0_;
    std::vector<std::complex<float>> h0_conj_;
    std::vector<float>               omega_;
    std::vector<std::complex<float>> ht_;
    std::vector<std::complex<float>> dx_spec_;
    std::vector<std::complex<float>> dz_spec_;
    std::vector<std::complex<float>> h_out_;
    std::vector<std::complex<float>> dx_out_;
    std::vector<std::complex<float>> dz_out_;

    kiss_fftnd_state* ifft_ = nullptr;

    void buildMesh_();
    void allocBuffers_();
    void computeInitialSpectrum_();
    float philips_(float kx, float kz) const;
};
