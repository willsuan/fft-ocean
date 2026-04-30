// Sanity tests for Ocean. No external test framework, just asserts and a
// running tally. Build target is ocean_tests in CMakeLists.txt.

#include "Ocean.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

static int    g_pass = 0;
static int    g_fail = 0;
static const char* g_curname = nullptr;

#define CHECK(cond, msg) do {                                          \
    if (cond) { ++g_pass; }                                            \
    else { ++g_fail;                                                   \
           std::fprintf(stderr, "FAIL [%s]: %s  (%s:%d)\n",            \
                        g_curname ? g_curname : "?",                   \
                        msg, __FILE__, __LINE__); }                    \
} while (0)

#define BEGIN(name) do { g_curname = (name);                           \
                         std::printf("-- %s\n", name); } while (0)

static bool finiteAll(const Eigen::MatrixXd& M) {
    for (int i = 0; i < M.size(); ++i)
        if (!std::isfinite(M.data()[i])) return false;
    return true;
}

static bool finiteAll(const Eigen::VectorXd& v) {
    for (int i = 0; i < v.size(); ++i)
        if (!std::isfinite(v(i))) return false;
    return true;
}

static double maxAbsHeight(const Eigen::MatrixXd& V) {
    double m = 0.0;
    for (int i = 0; i < V.rows(); ++i)
        m = std::max(m, std::fabs(V(i, 1)));
    return m;
}

static double sumAbsDiff(const Eigen::MatrixXd& A,
                         const Eigen::MatrixXd& B,
                         int col) {
    double s = 0.0;
    for (int i = 0; i < A.rows(); ++i)
        s += std::fabs(A(i, col) - B(i, col));
    return s;
}

int main() {
    // ------------------------------------------------------------------
    BEGIN("mesh dims");
    {
        Ocean::Params p;
        p.N = 64;
        p.tile = 2;
        Ocean ocean(p);
        int side = 2 * 64 + 1;
        CHECK(ocean.vertices().rows() == side * side, "vertex count");
        CHECK(ocean.vertices().cols() == 3,           "xyz cols");
        CHECK(ocean.faces().rows() == 2 * (side-1) * (side-1), "face count");
        CHECK(ocean.faces().cols() == 3,              "tri cols");
        CHECK(ocean.meshSide() == side,               "meshSide()");
        CHECK(ocean.foam().size() == side * side,     "foam matches mesh");
    }

    // ------------------------------------------------------------------
    BEGIN("update finiteness");
    {
        Ocean ocean;
        ocean.update(0.0f, 1.0f, 0.85f, 1.0f);
        CHECK(finiteAll(ocean.vertices()), "vertices finite at t=0");
        CHECK(finiteAll(ocean.foam()),     "foam finite at t=0");

        ocean.update(5.0f, 1.0f, 0.85f, 1.0f);
        CHECK(finiteAll(ocean.vertices()), "vertices finite at t=5");
        CHECK(finiteAll(ocean.foam()),     "foam finite at t=5");

        ocean.update(123.456f, 1.0f, 0.85f, 1.0f);
        CHECK(finiteAll(ocean.vertices()), "vertices finite at large t");
    }

    // ------------------------------------------------------------------
    BEGIN("height extents bounded");
    {
        Ocean ocean;
        ocean.update(1.0f, 0.8f, 0.85f, 1.0f);
        double m = maxAbsHeight(ocean.vertices());
        CHECK(m < 50.0, "|h| bounded under default params");
        CHECK(m > 0.0,  "h not all zero");
    }

    // ------------------------------------------------------------------
    BEGIN("foam range");
    {
        Ocean ocean;
        ocean.update(2.0f, 0.8f, 1.0f, 12.0f);
        const auto& f = ocean.foam();
        bool inRange = true;
        for (int i = 0; i < f.size(); ++i)
            if (f(i) < 0.0 || f(i) > 1.0) { inRange = false; break; }
        CHECK(inRange, "foam in [0,1]");
    }

    // ------------------------------------------------------------------
    BEGIN("time evolution");
    {
        Ocean ocean;
        ocean.update(0.0f, 1.0f, 0.85f, 1.0f);
        Eigen::MatrixXd V0 = ocean.vertices();
        ocean.update(3.0f, 1.0f, 0.85f, 1.0f);
        double dy = sumAbsDiff(V0, ocean.vertices(), 1);
        CHECK(dy > 1e-3, "heights change with time");
    }

    // ------------------------------------------------------------------
    BEGIN("displayGain linearity");
    {
        Ocean ocean;
        ocean.update(0.0f, 0.0f, 0.85f, 1.0f);
        Eigen::MatrixXd V1 = ocean.vertices();
        ocean.update(0.0f, 0.0f, 0.85f, 4.0f);
        Eigen::MatrixXd V4 = ocean.vertices();
        double maxAbs = 0.0;
        int    pick   = 0;
        for (int i = 0; i < V1.rows(); ++i) {
            if (std::fabs(V1(i, 1)) > maxAbs) {
                maxAbs = std::fabs(V1(i, 1));
                pick   = i;
            }
        }
        if (maxAbs < 1e-6) {
            CHECK(true, "skipped (flat)");
        } else {
            double ratio = V4(pick, 1) / V1(pick, 1);
            CHECK(std::fabs(ratio - 4.0) < 1e-3, "h scales by displayGain");
        }
    }

    // ------------------------------------------------------------------
    BEGIN("reseed determinism");
    {
        Ocean::Params p;
        p.layers.clear();
        Ocean a(p), b(p);
        a.update(1.5f, 1.0f, 0.85f, 1.0f);
        b.update(1.5f, 1.0f, 0.85f, 1.0f);
        double diff = sumAbsDiff(a.vertices(), b.vertices(), 1);
        CHECK(diff < 1e-6, "same seed -> same heights");
        double fdiff = 0.0;
        for (int i = 0; i < a.foam().size(); ++i)
            fdiff += std::fabs(a.foam()(i) - b.foam()(i));
        CHECK(fdiff < 1e-6, "same seed -> same foam");
    }

    // ------------------------------------------------------------------
    BEGIN("different seeds -> different oceans");
    {
        Ocean::Params p1; p1.layers.clear();
        Ocean::Params p2; p2.layers.clear();
        // build defaults, then bump one layer's seed
        Ocean a(p1);
        Ocean::Params p2b = a.params();
        p2b.layers[0].seed += 1;
        Ocean b(p2b);
        a.update(0.5f, 1.0f, 0.85f, 1.0f);
        b.update(0.5f, 1.0f, 0.85f, 1.0f);
        double diff = sumAbsDiff(a.vertices(), b.vertices(), 1);
        CHECK(diff > 1e-3, "changing seed changes the ocean");
    }

    // ------------------------------------------------------------------
    BEGIN("live weight edits");
    {
        Ocean ocean;
        ocean.update(0.5f, 1.0f, 0.85f, 1.0f);
        Eigen::MatrixXd before = ocean.vertices();
        ocean.setLayerWeight(0, 0.0f);
        ocean.update(0.5f, 1.0f, 0.85f, 1.0f);
        double diff = sumAbsDiff(before, ocean.vertices(), 1);
        CHECK(diff > 1e-3, "muting a layer changes the surface");
    }

    // ------------------------------------------------------------------
    BEGIN("muting all layers flattens surface");
    {
        Ocean ocean;
        for (size_t i = 0; i < ocean.params().layers.size(); ++i)
            ocean.setLayerWeight(i, 0.0f);
        ocean.update(2.0f, 1.0f, 0.85f, 1.0f);
        CHECK(maxAbsHeight(ocean.vertices()) < 1e-5, "all-muted -> h=0");
    }

    // ------------------------------------------------------------------
    BEGIN("zero choppiness leaves x,z on grid");
    {
        Ocean ocean;
        ocean.update(0.0f, 0.0f, 0.85f, 1.0f);
        const int  side = ocean.meshSide();
        const auto& V    = ocean.vertices();
        // vertex (I,J) should sit at world_x = I*dx - half regardless of h.
        // pick the centre vertex and a corner; their X/Z must match the grid.
        int center = (side / 2) * side + (side / 2);
        int corner = 0;
        // we expect |x - grid_x| = 0 because chop=0 -> Dx,Dz = 0
        // can't check exact world_x without knowing dxPhys here, but the
        // X column should at least be deterministic and finite.
        CHECK(std::isfinite(V(center, 0)) && std::isfinite(V(center, 2)),
              "centre vertex finite");
        CHECK(std::isfinite(V(corner, 0)) && std::isfinite(V(corner, 2)),
              "corner vertex finite");
        // and crucially: with chop=0, two runs at different displayGain
        // should leave x,z untouched (only y scales).
        Eigen::MatrixXd V1 = V;
        ocean.update(0.0f, 0.0f, 0.85f, 7.0f);
        double dx = sumAbsDiff(V1, ocean.vertices(), 0);
        double dz = sumAbsDiff(V1, ocean.vertices(), 2);
        CHECK(dx < 1e-6, "x untouched when chop=0");
        CHECK(dz < 1e-6, "z untouched when chop=0");
    }

    // ------------------------------------------------------------------
    BEGIN("choppiness perturbs x,z");
    {
        Ocean ocean;
        ocean.update(1.0f, 0.0f, 0.85f, 1.0f);
        Eigen::MatrixXd V0 = ocean.vertices();
        ocean.update(1.0f, 1.0f, 0.85f, 1.0f);
        double dx = sumAbsDiff(V0, ocean.vertices(), 0);
        double dz = sumAbsDiff(V0, ocean.vertices(), 2);
        CHECK(dx > 1e-3, "x shifts under chop>0");
        CHECK(dz > 1e-3, "z shifts under chop>0");
    }

    // ------------------------------------------------------------------
    BEGIN("higher choppiness -> stronger horizontal displacement");
    {
        Ocean ocean;
        ocean.update(1.0f, 0.5f, 0.85f, 1.0f);
        Eigen::MatrixXd Vlow = ocean.vertices();
        ocean.update(1.0f, 1.5f, 0.85f, 1.0f);
        Eigen::MatrixXd Vhigh = ocean.vertices();
        // displacement magnitude should roughly track choppiness
        double sumLow  = 0.0, sumHigh = 0.0;
        // baseline grid was the same; deviation from grid = displacement.
        // we don't know the grid, but the deviation across the field
        // should grow.
        for (int i = 0; i < Vlow.rows(); ++i) {
            sumLow  += std::fabs(Vlow(i, 0))  + std::fabs(Vlow(i, 2));
            sumHigh += std::fabs(Vhigh(i, 0)) + std::fabs(Vhigh(i, 2));
        }
        CHECK(sumHigh > sumLow, "higher chop -> larger |x|+|z|");
    }

    // ------------------------------------------------------------------
    BEGIN("foam stays empty at very low threshold");
    {
        Ocean ocean;
        // threshold below the lowest crossfade boundary -> foam should be
        // essentially zero everywhere
        ocean.update(0.0f, 0.5f, -2.0f, 1.0f);
        double maxFoam = 0.0;
        for (int i = 0; i < ocean.foam().size(); ++i)
            maxFoam = std::max(maxFoam, ocean.foam()(i));
        CHECK(maxFoam < 1e-5, "very low threshold -> no foam");
    }

    // ------------------------------------------------------------------
    BEGIN("foam saturates at high threshold");
    {
        Ocean ocean;
        ocean.update(0.0f, 0.5f, 5.0f, 1.0f);
        double minFoam = 1.0;
        for (int i = 0; i < ocean.foam().size(); ++i)
            minFoam = std::min(minFoam, ocean.foam()(i));
        CHECK(minFoam > 0.99, "very high threshold -> foam everywhere");
    }

    // ------------------------------------------------------------------
    BEGIN("tile count change resizes mesh");
    {
        Ocean::Params p;
        p.N = 32; p.tile = 1;
        Ocean ocean(p);
        int side1 = ocean.meshSide();

        Ocean::Params q = ocean.params();
        q.tile = 4;
        ocean.reseed(q);
        int side4 = ocean.meshSide();

        CHECK(side4 == 4 * side1 - 3,             "tile 4 = 4*N+1");
        CHECK(ocean.vertices().rows() == side4 * side4,
              "vertex count grows with tile");
        CHECK(ocean.faces().rows() == 2 * (side4-1) * (side4-1),
              "face count grows with tile");
    }

    // ------------------------------------------------------------------
    BEGIN("custom layers respected");
    {
        Ocean::Params p;
        Ocean::Layer onlyLayer;
        onlyLayer.L          = 200.f;
        onlyLayer.windSpeed  = 12.f;
        onlyLayer.windDirDeg = 45.f;
        onlyLayer.amplitude  = 0.5f;
        onlyLayer.weight     = 1.0f;
        onlyLayer.seed       = 42;
        p.layers = { onlyLayer };
        Ocean ocean(p);
        CHECK(ocean.params().layers.size() == 1u, "single-layer ocean");
        ocean.update(1.0f, 0.8f, 0.85f, 1.0f);
        CHECK(finiteAll(ocean.vertices()),        "single-layer finite");
    }

    // ------------------------------------------------------------------
    BEGIN("foam smoother after blur (low spatial freq)");
    {
        // after the spatial blur pass, neighbouring foam values shouldn't
        // jump by more than ~0.5 across one vertex spacing.
        Ocean ocean;
        ocean.update(2.0f, 1.0f, 1.1f, 12.0f);
        const int side = ocean.meshSide();
        const auto& f  = ocean.foam();
        double maxJump = 0.0;
        for (int I = 1; I < side - 1; ++I) {
            for (int J = 1; J < side - 1; ++J) {
                double v = f(I * side + J);
                double a = f((I-1) * side + J);
                double b = f((I+1) * side + J);
                double c = f(I * side + (J-1));
                double d = f(I * side + (J+1));
                maxJump = std::max(maxJump, std::fabs(v - a));
                maxJump = std::max(maxJump, std::fabs(v - b));
                maxJump = std::max(maxJump, std::fabs(v - c));
                maxJump = std::max(maxJump, std::fabs(v - d));
            }
        }
        CHECK(maxJump < 0.5, "no per-vertex foam steps > 0.5");
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
