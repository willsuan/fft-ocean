# FFT Ocean

CS 384P final project. Real-time FFT ocean surface using Tessendorf 2001
with a 3-layer cascade, choppy displacement, and Jacobian foam.

## Layout

- `src/` - Ocean.h, Ocean.cpp, main.cpp
- `tests/test_ocean.cpp` - sanity test suite
- `CMakeLists.txt` - cross-platform build, fetches deps automatically
- `report/report.pdf` - writeup
- `report/figs/` - screenshots used in the writeup
- `demo.mp4` - narrated demo video

Dependencies (Polyscope, Eigen, kissfft) are pulled by CMake FetchContent
on first configure, so no system packages are needed.

## Build

Requires CMake 3.20+ and a C++17 compiler.

macOS / Linux:
```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
./bin/ocean
```

Optional - run the test suite:
```
cmake --build . --target ocean_tests
./bin/ocean_tests
```

Windows (Visual Studio):
```
mkdir build
cd build
cmake ..
cmake --build . --config Release
.\bin\Release\ocean.exe
```

The first configure pulls Polyscope, Eigen and kissfft from upstream git;
later builds are incremental.

## Running it

Left-mouse drag to orbit, scroll to zoom. The ImGui panel on the right
exposes:

- play/pause and a time scale slider
- per-layer wind speed, direction, patch size L, amplitude, mix weight
- choppiness (0 = pure sines, 2 = sharp Gerstner-style crests)
- vertical gain (cosmetic, multiplies the rendered heightfield)
- foam toggle and the Jacobian threshold for foam onset
- 4 palette presets (Ocean / Sunset / Lagoon / Storm)
- tile count and tile size for the rendered grid

For the showiest foam: hit the Storm preset, push wind speed to 35,
choppiness to 2.0.

## Survey

I have submitted the online course instructor survey for CS 384P
(Sp 2026).

## Notes

Solo project, no collaboration report. No starter code was used.
External libraries are unmodified.
