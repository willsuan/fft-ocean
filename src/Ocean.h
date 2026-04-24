// Ocean.h — Tessendorf FFT ocean with a multi-scale patch cascade.
//
// Based on Jerry Tessendorf, "Simulating Ocean Water" (SIGGRAPH 2001 course
// notes). Each internal patch is a standalone FFT synthesis of the Philips
// spectrum at some scale (patch length L, wind speed, amplitude); the final
// surface is the sum of several such patches evaluated at the same world
// coordinates. Because different patches have different spatial periods, the
// composite is visually non-repeating even though each layer is periodic —
// the standard trick used by AAA game engines to hide FFT tiling.
//
// Features:
//  - Philips wind-wave spectrum with directional preference + upwind damping
//  - Complex Gaussian initial amplitudes (Box–Muller)
//  - Real-preserving time evolution
//  - Choppy waves (horizontal displacement) via two extra IFFTs per patch
//  - Height-based foam proxy
//  - Seamless K×K tiling of the simulated domain
//  - Multi-scale cascade (default 3 layers: swell / chop / ripple)
#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>

struct kiss_fftnd_state;

class Ocean {
public:
    // Parameters for a single cascade layer.
    struct Layer {
        float L          = 250.0f;   // patch size (meters)
        float windSpeed  = 20.0f;
        float amplitude  = 1.5f;
        float weight     = 1.0f;     // output scaling, for solo/mute in UI
        unsigned seed    = 1337;
    };

    struct Params {
        int   N           = 128;
        int   tile        = 3;       // K: render K×K tiled copies
        float windDirDeg  = 0.0f;    // shared across all layers (same wind)
        float gravity     = 9.81f;
        float cutoff      = 0.001f;
        std::vector<Layer> layers;   // cascade; if empty, populated w/ 3 defaults
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
    // Spectrum state for one cascade layer.
    struct PatchState {
        Layer                            layer;
        std::vector<std::complex<float>> h0;
        std::vector<std::complex<float>> h0_conj;
        std::vector<float>               omega;
        std::vector<std::complex<float>> ht;
        std::vector<std::complex<float>> dx_spec;
        std::vector<std::complex<float>> dz_spec;
        std::vector<std::complex<float>> h_out;
        std::vector<std::complex<float>> dx_out;
        std::vector<std::complex<float>> dz_out;
    };

    Params p_;
    std::vector<PatchState> patches_;

    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    Eigen::VectorXd foam_;

    kiss_fftnd_state* ifft_ = nullptr;

    void buildMesh_();
    void allocBuffersAndSpectrum_();
    void initLayer_(PatchState& s);
    float philips_(const Layer& layer, float kx, float kz) const;
};
