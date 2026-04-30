// main.cpp - polyscope viewer + ImGui controls for the FFT ocean.

#include "Ocean.h"

#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/view.h>

#include <chrono>
#include <limits>

static Ocean* g_ocean = nullptr;
static polyscope::SurfaceMesh* g_mesh = nullptr;

static bool  g_playing       = true;
static float g_time          = 0.0f;
static float g_speed         = 1.0f;
static float g_vgain         = 12.0f;     // visual height multiplier (cosmetic)
static float g_choppiness    = 0.8f;
static float g_foamThreshold = 0.5f;      // foam shows where J < this
static bool  g_showFoam      = true;

// water palette: deep -> mid -> crest, plus foam color
static float g_deep[3]  = {0.02f, 0.12f, 0.28f};
static float g_mid[3]   = {0.10f, 0.40f, 0.60f};
static float g_crest[3] = {0.55f, 0.85f, 0.95f};
static float g_foamCol[3] = {1.0f, 1.0f, 1.0f};

static Ocean::Params g_params;

static void callback() {
    ImGui::PushItemWidth(160);
    ImGui::Text("FFT Ocean (Tessendorf, cascade)");
    ImGui::Separator();

    ImGui::Checkbox("Playing", &g_playing);
    ImGui::SliderFloat("Time scale", &g_speed, 0.0f, 3.0f);
    ImGui::Text("Sim time: %.2f s", g_time);

    ImGui::Separator();
    ImGui::Text("Global");
    bool dirty = false;
    int  prevTile = g_params.tile;
    dirty |= ImGui::SliderInt  ("Tile count", &g_params.tile, 1, 7);
    dirty |= ImGui::SliderFloat("Tile size (m)", &g_params.tileSize, 100.0f, 1000.0f);
    if (ImGui::Button("Reseed all")) {
        for (auto& layer : g_params.layers) layer.seed++;
        dirty = true;
    }

    ImGui::Separator();
    ImGui::Text("Cascade layers");
    for (size_t i = 0; i < g_params.layers.size(); ++i) {
        ImGui::PushID(static_cast<int>(i));
        ImGui::Text("Layer %zu (L=%.0fm)", i, g_params.layers[i].L);
        dirty |= ImGui::SliderFloat("patch L",    &g_params.layers[i].L,          2.0f, 600.0f);
        dirty |= ImGui::SliderFloat("wind speed", &g_params.layers[i].windSpeed,  1.0f, 50.0f);
        dirty |= ImGui::SliderFloat("wind dir",   &g_params.layers[i].windDirDeg, 0.0f, 360.0f);
        dirty |= ImGui::SliderFloat("amplitude",  &g_params.layers[i].amplitude,  1e-3f, 10.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("weight", &g_params.layers[i].weight, 0.0f, 2.0f);  // weight is live, no rebuild
        ImGui::Separator();
        ImGui::PopID();
    }
    if (dirty) {
        g_ocean->reseed(g_params);
        // tile count change resizes the mesh, so re-register or polyscope blows up
        if (g_params.tile != prevTile) {
            g_mesh = polyscope::registerSurfaceMesh("ocean",
                                                    g_ocean->vertices(),
                                                    g_ocean->faces());
            g_mesh->setSmoothShade(true);
        }
    }

    ImGui::Text("Waves");
    ImGui::SliderFloat("Choppiness",   &g_choppiness, 0.0f, 1.5f);
    ImGui::SliderFloat("Vertical gain", &g_vgain, 1.0f, 100.0f, "%.1fx");
    ImGui::Checkbox("Show foam", &g_showFoam);
    ImGui::SliderFloat("Foam threshold (Jacobian)", &g_foamThreshold, 0.0f, 0.9f);

    ImGui::Separator();
    ImGui::Text("Palette");
    ImGui::ColorEdit3("Deep",  g_deep);
    ImGui::ColorEdit3("Mid",   g_mid);
    ImGui::ColorEdit3("Crest", g_crest);
    ImGui::ColorEdit3("Foam",  g_foamCol);
    if (ImGui::Button("Preset: Ocean"))   { g_deep[0]=0.02f; g_deep[1]=0.12f; g_deep[2]=0.28f;  g_mid[0]=0.10f; g_mid[1]=0.40f; g_mid[2]=0.60f;  g_crest[0]=0.55f; g_crest[1]=0.85f; g_crest[2]=0.95f;  g_foamCol[0]=1.0f; g_foamCol[1]=1.0f; g_foamCol[2]=1.0f; }
    ImGui::SameLine();
    if (ImGui::Button("Preset: Sunset"))  { g_deep[0]=0.25f; g_deep[1]=0.05f; g_deep[2]=0.20f;  g_mid[0]=0.85f; g_mid[1]=0.40f; g_mid[2]=0.30f;  g_crest[0]=1.00f; g_crest[1]=0.80f; g_crest[2]=0.55f;  g_foamCol[0]=1.0f; g_foamCol[1]=0.9f; g_foamCol[2]=0.7f; }
    ImGui::SameLine();
    if (ImGui::Button("Preset: Lagoon"))  { g_deep[0]=0.00f; g_deep[1]=0.35f; g_deep[2]=0.45f;  g_mid[0]=0.15f; g_mid[1]=0.70f; g_mid[2]=0.70f;  g_crest[0]=0.65f; g_crest[1]=0.95f; g_crest[2]=0.90f;  g_foamCol[0]=1.0f; g_foamCol[1]=1.0f; g_foamCol[2]=1.0f; }
    ImGui::SameLine();
    if (ImGui::Button("Preset: Storm"))   { g_deep[0]=0.04f; g_deep[1]=0.06f; g_deep[2]=0.09f;  g_mid[0]=0.20f; g_mid[1]=0.25f; g_mid[2]=0.30f;  g_crest[0]=0.55f; g_crest[1]=0.60f; g_crest[2]=0.65f;  g_foamCol[0]=0.95f; g_foamCol[1]=0.97f; g_foamCol[2]=1.0f; }

    ImGui::PopItemWidth();

    if (g_playing) {
        static auto last = std::chrono::steady_clock::now();
        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;
        g_time += dt * g_speed;
    }

    // push live weight edits straight through (no full rebuild)
    for (size_t i = 0; i < g_params.layers.size(); ++i) {
        g_ocean->setLayerWeight(i, g_params.layers[i].weight);
    }

    // pass vgain into the simulator so chop displacement scales with height
    g_ocean->update(g_time, g_choppiness, g_foamThreshold, g_vgain);

    Eigen::MatrixXd V = g_ocean->vertices();
    g_mesh->updateVertexPositions(V);

    // shade the mesh ourselves: polyscope's smooth shading averages too much
    // and you lose the wave detail at distance. analytic lambert on the
    // gained heightfield's slopes keeps crests bright and troughs dark.
    const int NV   = V.rows();
    const int side = g_ocean->meshSide();
    Eigen::MatrixXd C(NV, 3);
    const Eigen::Vector3d deep {g_deep[0],  g_deep[1],  g_deep[2]};
    const Eigen::Vector3d mid  {g_mid[0],   g_mid[1],   g_mid[2]};
    const Eigen::Vector3d crest{g_crest[0], g_crest[1], g_crest[2]};
    const Eigen::Vector3d white{g_foamCol[0], g_foamCol[1], g_foamCol[2]};

    double hMin =  std::numeric_limits<double>::infinity();
    double hMax = -std::numeric_limits<double>::infinity();
    for (int v = 0; v < NV; ++v) {
        double h = V(v, 1);
        if (h < hMin) hMin = h;
        if (h > hMax) hMax = h;
    }
    const double hRange = std::max(hMax - hMin, 1e-6);

    const Eigen::Vector3d lightDir = Eigen::Vector3d(0.3, 0.85, 0.45).normalized();
    const double ambient = 0.35;

    const double dxPhys = static_cast<double>(g_ocean->params().tileSize)
                          / static_cast<double>(g_ocean->params().N);

    auto idx = [&](int I, int J) { return I * side + J; };
    auto sampleH = [&](int I, int J) {
        I = std::max(0, std::min(side - 1, I));
        J = std::max(0, std::min(side - 1, J));
        return V(idx(I, J), 1);
    };

    const Eigen::VectorXd& foam = g_ocean->foam();
    for (int I = 0; I < side; ++I) {
        for (int J = 0; J < side; ++J) {
            int v = idx(I, J);
            // central differences for the slope -> normal
            double sx = (sampleH(I+1, J) - sampleH(I-1, J)) / (2.0 * dxPhys);
            double sz = (sampleH(I, J+1) - sampleH(I, J-1)) / (2.0 * dxPhys);
            Eigen::Vector3d N(-sx, 1.0, -sz);
            N.normalize();
            double lambert = ambient + (1.0 - ambient) * std::max(0.0, N.dot(lightDir));

            // height ramp deep -> mid -> crest
            double t = (V(v, 1) - hMin) / hRange;
            Eigen::Vector3d base = (t < 0.5)
                ? (1.0 - 2.0 * t) * deep + (2.0 * t) * mid
                : (2.0 - 2.0 * t) * mid + (2.0 * t - 1.0) * crest;

            // foam factor as-is; no gamma boost (it was over-amplifying)
            double f = g_showFoam ? std::max(0.0, foam(v)) : 0.0;
            // foam stays ~white regardless of lighting; only the water gets lit
            Eigen::Vector3d shaded = lambert * (1.0 - f) * base + f * white;
            C.row(v) = shaded.cwiseMin(1.0).cwiseMax(0.0);
        }
    }
    g_mesh->addVertexColorQuantity("water color", C)->setEnabled(true);
}

int main() {
    polyscope::options::programName = "FFT Ocean";
    polyscope::options::verbosity   = 0;
    polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::None;
    polyscope::init();

    g_params.N = 128;
    g_params.tile = 3;
    Ocean ocean(g_params);
    g_ocean = &ocean;

    // mirror the layers Ocean filled in so the UI has something to bind to
    g_params.layers = ocean.params().layers;

    g_mesh = polyscope::registerSurfaceMesh("ocean", ocean.vertices(), ocean.faces());
    g_mesh->setSmoothShade(true);

    polyscope::view::lookAt(glm::vec3(500.f, 250.f, 500.f), glm::vec3(0.f, 0.f, 0.f));

    polyscope::state::userCallback = callback;
    polyscope::show();
    return 0;
}
