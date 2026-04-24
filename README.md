# FFT Ocean

A Tessendorf-style FFT ocean heightfield simulation for the CS 384P final project.

## Build

Requires CMake >= 3.20 and a C++17 compiler. Dependencies (Polyscope, Eigen,
GLFW, ImGui, kissfft) are fetched automatically by CMake — no system packages
needed.

### macOS / Linux
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
./bin/ocean
```

### Windows (Visual Studio)
```powershell
mkdir build; cd build
cmake ..
cmake --build . --config Release
.\bin\Release\ocean.exe
```

## Status

**Day 1 (scaffold):** Polyscope viewer with an animated sine-wave heightfield,
proving the per-frame mesh-update loop works. The real Tessendorf IFFT
evolution lands in Day 3.

## Roadmap

- [x] Day 1–2: CMake scaffold + animated heightfield viewer
- [ ] Day 3–5: Philips spectrum + IFFT time evolution (core Tessendorf)
- [ ] Day 6–7: Choppy waves (horizontal displacement) + whitecaps
- [ ] Day 8–9: Skybox + interactive wind controls + polish
- [ ] Day 10: Render portfolio clips
- [ ] Day 11: Writeup
- [ ] Day 12: Buffer + submit
