# Komorebi (KMRB)

A GPU particle simulation editor built with Vulkan 1.3. Write compute shaders, drag them onto entities, and tweak parameters in real-time through auto-generated UI controls.

The engine reads your compiled SPIR-V shaders via SPIRV-Reflect to discover push constant variables, then builds Inspector sliders for each one. Change a value, see it on the GPU next frame. Edit a shader file externally, and the engine hot-reloads it automatically.

## What it does

- **Pipeline entities** own three shader slots (compute, vertex, fragment). Drag files from the Project Browser onto the Inspector to assign them.
- **Shader reflection** extracts push constant parameters and generates live-tweakable controls (floats, vectors, ints, bools).
- **Hot-reload** watches shader files on disk and recompiles when they change.
- **Data Output** panel reads back SSBO data into a scrollable table with configurable refresh rate.
- **CSV export** for particle data snapshots.
- **ECS** powered by EnTT — entities (Pipeline, Camera, Grid) with components shown in a scene hierarchy and context-sensitive Inspector.

## Prerequisites

- Windows 10/11
- [Vulkan SDK](https://vulkan.lunarg.com/) 1.3+ (make sure `VULKAN_SDK` env var is set)
- [CMake](https://cmake.org/) 3.20+
- [vcpkg](https://github.com/microsoft/vcpkg)
- Visual Studio 2022 or any C++20 MSVC toolchain

## vcpkg dependencies

```
vcpkg install glfw3 glm entt
```

## Build & Run

```
git clone https://github.com/finn7199/Komorebi.git
cd Komorebi

cmake -B build -DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%/scripts/buildsystems/vcpkg.cmake
cmake --build build --config Debug
```

```
./build/Debug/Komorebi.exe
```

Or open `build/Komorebi.sln` in Visual Studio and hit F5.

## Writing shaders

New compute shaders can be created from File > New Compute Shader. The template includes push constant instructions:

```glsl
layout(push_constant) uniform Params {
    mat4 model;       // Engine built-in (do not remove)
    vec4 color;       // Engine built-in (do not remove)

    // Your parameters below - these become Inspector sliders
    float gravity;
    float damping;
} params;
```

`model` and `color` (offsets 0-79) are reserved by the engine. Your parameters go after, up to 128 bytes total.

## Tech stack

Vulkan 1.3, Dear ImGui (docking), EnTT, GLM, GLFW, glslc, SPIRV-Reflect
