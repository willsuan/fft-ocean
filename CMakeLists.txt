cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0169 OLD)

project("FFT Ocean")

# Prevent in-source builds
if (CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    message(FATAL_ERROR "Source and build directories cannot be the same. Use a /build directory.")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

### Output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

### Compiler flags (matches milestone2 convention)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(BASE_CXX_FLAGS "-std=c++17 -Wall -Wextra -g3")
    set(DISABLED_WARNINGS " -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-sign-compare")
    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=5 -fcolor-diagnostics")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
        set(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
    endif()
    set(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "-march=native -O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    set(CMAKE_CXX_STANDARD 17)
    set(BASE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /MP /bigobj")
    set(DISABLED_WARNINGS "/wd\"4267\" /wd\"4244\" /wd\"4305\"")
    set(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
    add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
    add_definitions(-DNOMINMAX)
    add_definitions(-D_USE_MATH_DEFINES)
else()
    message(FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]")
endif()

include(FetchContent)

### Eigen (header-only)
message("\n== Fetching Eigen ==")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

### Polyscope
message("\n== Fetching Polyscope ==")
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG v2.3.0
)
FetchContent_MakeAvailable(polyscope)

### kissfft — header-configurable, cross-platform FFT
message("\n== Fetching kissfft ==")
set(KISSFFT_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(KISSFFT_STATIC ON CACHE BOOL "" FORCE)
set(KISSFFT_TEST OFF CACHE BOOL "" FORCE)
set(KISSFFT_TOOLS OFF CACHE BOOL "" FORCE)
set(KISSFFT_USE_ALLOCA OFF CACHE BOOL "" FORCE)
set(KISSFFT_DATATYPE "float" CACHE STRING "" FORCE)
FetchContent_Declare(
    kissfft
    GIT_REPOSITORY https://github.com/mborgerding/kissfft.git
    GIT_TAG 131.1.0
)
FetchContent_MakeAvailable(kissfft)

### Executable
add_executable(ocean
    src/main.cpp
    src/Ocean.cpp
)

target_include_directories(ocean PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${kissfft_SOURCE_DIR}
)

target_link_libraries(ocean PUBLIC
    polyscope
    Eigen3::Eigen
    kissfft
)
