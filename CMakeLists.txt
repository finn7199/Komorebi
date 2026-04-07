cmake_minimum_required(VERSION 3.20)
project(Komorebi)

set(CMAKE_CXX_STANDARD 20)

# Find our libraries in the vcpkg warehouse
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(Stb REQUIRED)

# Vulkan SDK ships shaderc and SPIRV tools — point CMake to them
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")

# ── Dear ImGui (vendored docking branch) ──
set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/external/imgui_docking")
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(imgui PUBLIC Vulkan::Vulkan glfw)

add_executable(Komorebi
    src/main.cpp
    src/kmrb_core.cpp
    src/kmrb_core.hpp
    src/kmrb_renderer.cpp
    src/kmrb_renderer.hpp
    src/kmrb_sim.cpp
    src/kmrb_sim.hpp
    src/kmrb_buffers.cpp
    src/kmrb_buffers.hpp
    src/kmrb_ui.cpp
    src/kmrb_ui.hpp
    src/kmrb_camera.cpp
    src/kmrb_camera.hpp
    src/kmrb_types.hpp
    src/stb_image_impl.cpp
)

# Link the libraries to our executable
target_link_libraries(Komorebi PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui
    EnTT::EnTT
)

# Pass source paths to code so shaders can be found regardless of working directory
target_compile_definitions(Komorebi PRIVATE
    KMRB_SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders"
    KMRB_SHADER_SPV_DIR="${CMAKE_BINARY_DIR}/shaders"
)

# Vulkan SDK include path (for spirv-reflect, spirv-cross headers)
target_include_directories(Komorebi PRIVATE
    "${VULKAN_SDK_PATH}/Include"
    "${VULKAN_SDK_PATH}/Source/SPIRV-Reflect"
    ${Stb_INCLUDE_DIR}
)

# Vulkan SDK libraries: spirv-reflect (introspection)
# Note: shaderc is used via glslc subprocess, not linked directly (avoids Debug/Release mismatch)
target_link_libraries(Komorebi PRIVATE
    "${VULKAN_SDK_PATH}/Lib/spirv-reflect-static.lib"
)

# ── Shader Compilation (GLSL → SPIR-V via glslc) ──
find_program(GLSLC glslc HINTS "${VULKAN_SDK_PATH}/Bin")
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB_RECURSE SHADERS "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp")

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    # All SPV outputs go flat into SHADER_OUT_DIR regardless of source subdir
    set(SPIRV_OUTPUT "${SHADER_OUT_DIR}/${SHADER_NAME}.spv")
    add_custom_command(
        OUTPUT ${SPIRV_OUTPUT}
        COMMAND ${GLSLC} ${SHADER} -o ${SPIRV_OUTPUT}
        DEPENDS ${SHADER}
        COMMENT "Compiling shader: ${SHADER_NAME}"
    )
    list(APPEND SPIRV_OUTPUTS ${SPIRV_OUTPUT})
endforeach()

add_custom_target(Shaders DEPENDS ${SPIRV_OUTPUTS})
add_dependencies(Komorebi Shaders)