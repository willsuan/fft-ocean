// main.cpp — FFT Ocean viewer entrypoint.

#include "Ocean.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/view.h>

#include <chrono>

static Ocean* g_ocean = nullptr;
static polyscope::SurfaceMesh* g_mesh = nullptr;

static bool  g_playing    = true;
static float g_time       = 0.0f;
static float g_speed      = 1.0f;
static float g_vgain      = 35.0f;
static float g_choppiness = 1.5f;
static float g_foamThreshold = 0.8f;
static bool  g_showFoam   = true;

static Ocean::Params g_params;

static void callback() {
    ImGui::PushItemWidth(160);
    ImGui::Text("FFT Ocean (Tessendorf)");
    ImGui::Separator();

    ImGui::Checkbox("Playing", &g_playing);
    ImGui::SliderFloat("Time scale", &g_speed, 0.0f, 3.0f);
    ImGui::Text("Sim time: %.2f s", g_time);

    ImGui::Separator();
    ImGui::Text("Spectrum");
    bool dirty = false;
    dirty |= ImGui::SliderFloat("Wind speed (m/s)", &g_params.windSpeed, 1.0f, 50.0f);
    dirty |= ImGui::SliderFloat("Wind dir (deg)",   &g_params.windDirDeg, 0.0f, 360.0f);
    dirty |= ImGui::SliderFloat("Amplitude", &g_params.amplitude, 1e-5f, 10.0f, "%.5f", ImGuiSliderFlags_Logarithmic);
    dirty |= ImGui::SliderFloat("Patch size (m)",   &g_params.L, 20.0f, 800.0f);
    dirty |= ImGui::SliderInt  ("Tile count",       &g_params.tile, 1, 7);
    if (ImGui::Button("Reseed")) { g_params.seed++; dirty = true; }
    if (dirty) g_ocean->reseed(g_params);

    ImGui::Separator();
    ImGui::Text("Waves");
    ImGui::SliderFloat("Choppiness",   &g_choppiness, 0.0f, 2.5f);
    ImGui::SliderFloat("Vertical gain", &g_vgain, 1.0f, 100.0f, "%.1fx");
    ImGui::Checkbox("Show foam", &g_showFoam);
    ImGui::SliderFloat("Foam threshold (crest %)", &g_foamThreshold, 0.0f, 0.99f);

    ImGui::PopItemWidth();

    if (g_playing) {
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        g_time += dt * g_speed;
    }

    g_ocean->update(g_time, g_choppiness, g_foamThreshold);

    Eigen::MatrixXd V = g_ocean->vertices();
    V.col(1) *= g_vgain;
    g_mesh->updateVertexPositions(V);

    // Per-vertex color: three-stop gradient (deep → mid → light) driven by
    // normalized height, with foam (white) layered on top. The height gradient
    // alone makes waves visually readable even when actual vertical relief is
    // modest, which matters because heightfield normals are subtle at a
    // distance.
    const int NV = V.rows();
    Eigen::MatrixXd C(NV, 3);
    const Eigen::Vector3d deep {0.02, 0.12, 0.28};  // trough
    const Eigen::Vector3d mid  {0.10, 0.40, 0.60};  // average
    const Eigen::Vector3d crest{0.55, 0.85, 0.95};  // crest (light cyan)
    const Eigen::Vector3d white{1.00, 1.00, 1.00};

    double hMin =  std::numeric_limits<double>::infinity();
    double hMax = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < NV; ++v) {
        double h = V(v, 1);
        if (h < hMin) hMin = h;
        if (h > hMax) hMax = h;
    }
    const double hRange = std::max(hMax - hMin, 1e-6);

    const Eigen::VectorXd& foam = g_ocean->foam();
    for (int v = 0; v < NV; ++v) {
        double t = (V(v, 1) - hMin) / hRange;
        Eigen::Vector3d base = (t < 0.5)
            ? (1.0 - 2.0 * t) * deep + (2.0 * t) * mid
            : (2.0 - 2.0 * t) * mid + (2.0 * t - 1.0) * crest;
        double f = g_showFoam ? foam(v) : 0.0;
        C.row(v) = (1.0 - f) * base + f * white;
    }
    g_mesh->addVertexColorQuantity("water color", C)->setEnabled(true);
}

int main() {
    polyscope::options::programName = "FFT Ocean";
    polyscope::options::verbosity   = 0;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::init();

    g_params.N = 128;
    g_params.L = 250.0f;
    g_params.tile = 3;
    Ocean ocean(g_params);
    g_ocean = &ocean;

    g_mesh = polyscope::registerSurfaceMesh("ocean", ocean.vertices(), ocean.faces());
    g_mesh->setSurfaceColor({0.10f, 0.35f, 0.55f});
    g_mesh->setSmoothShade(true);

    // Nudge the camera to a decent 3/4 view above the surface.
    polyscope::view::lookAt(glm::vec3(500.f, 250.f, 500.f), glm::vec3(0.f, 0.f, 0.f));

    polyscope::state::userCallback = callback;
    polyscope::show();
    return 0;
}
