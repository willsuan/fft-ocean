// Ocean.h - FFT ocean simulation, Tessendorf 2001 with a 3-layer cascade.
#pragma once

#include <Eigen/Core>
#include <complex>
#include <vector>

struct kiss_fftnd_state;

class Ocean {
public:
    // one cascade layer: an independent FFT ocean at scale L
    struct Layer {
        float L          = 250.0f;   // patch size in meters
        float windSpeed  = 20.0f;
        float windDirDeg = 0.0f;
        float amplitude  = 1.5f;
        float weight     = 1.0f;     // mix weight in the final sum
        unsigned seed    = 1337;
    };

    struct Params {
        int   N           = 128;        // grid size, must be power of 2
        int   tile        = 3;          // K in K x K tile grid
        float tileSize    = 400.0f;     // size of one tile (m)
        float gravity     = 9.81f;
        float cutoff      = 0.001f;     // small-wave cutoff factor
        std::vector<Layer> layers;      // empty = use defaults
    };

    Ocean() : Ocean(Params{}) {}
    explicit Ocean(const Params& p);
    ~Ocean();

    void reseed(const Params& p);
    // displayGain scales h, Dx and Dz uniformly so the rendered surface
    // stays geometrically self-consistent under visual exaggeration.
    void update(float t, float choppiness, float foamThreshold,
                float displayGain = 1.0f);

    void setLayerWeight(size_t i, float w);

    const Eigen::MatrixXd& vertices() const { return V_; }
    const Eigen::MatrixXi& faces()    const { return F_; }
    const Eigen::VectorXd& foam()     const { return foam_; }

    int meshSide() const { return p_.tile * p_.N + 1; }

    const Params& params() const { return p_; }

private:
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
        // real-valued fields after IFFT + sign flip + normalization
        std::vector<float>               h_real;
        std::vector<float>               dx_real;
        std::vector<float>               dz_real;
    };

    Params p_;
    std::vector<PatchState> patches_;

    Eigen::MatrixXd V_;
    Eigen::MatrixXi F_;
    Eigen::VectorXd foam_;

    std::vector<float> dxTotal_;
    std::vector<float> dzTotal_;

    kiss_fftnd_state* ifft_ = nullptr;

    void buildMesh_();
    void allocBuffersAndSpectrum_();
    void initLayer_(PatchState& s);
    float philips_(const Layer& layer, float kx, float kz) const;
};
