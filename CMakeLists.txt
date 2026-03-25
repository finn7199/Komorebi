cmake_minimum_required(VERSION 3.20)
project(Komorebi)

set(CMAKE_CXX_STANDARD 20)

# Find our libraries in the vcpkg warehouse
find_package(Vulkan REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(EnTT CONFIG REQUIRED)

# Vulkan SDK ships shaderc and SPIRV tools — point CMake to them
set(VULKAN_SDK_PATH "$ENV{VULKAN_SDK}")

add_executable(Komorebi
    src/main.cpp
    src/kmrb_core.cpp
    src/kmrb_core.hpp
    src/kmrb_renderer.cpp
    src/kmrb_renderer.hpp
    src/kmrb_types.hpp
)

# Link the libraries to our executable
target_link_libraries(Komorebi PRIVATE
    Vulkan::Vulkan
    glfw
    glm::glm
    imgui::imgui
    EnTT::EnTT
)

# Vulkan SDK include path (for shaderc, spirv-reflect, spirv-cross headers)
target_include_directories(Komorebi PRIVATE
    "${VULKAN_SDK_PATH}/Include"
)

# Vulkan SDK libraries: shaderc (runtime compilation) + spirv-reflect (introspection)
target_link_libraries(Komorebi PRIVATE
    "${VULKAN_SDK_PATH}/Lib/shaderc_combined.lib"
    "${VULKAN_SDK_PATH}/Lib/spirv-reflect-static.lib"
)

# ── Shader Compilation (GLSL → SPIR-V via glslc) ──
find_program(GLSLC glslc HINTS "${VULKAN_SDK_PATH}/Bin")
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/shaders")
file(MAKE_DIRECTORY ${SHADER_OUT_DIR})

file(GLOB SHADERS "${SHADER_DIR}/*.vert" "${SHADER_DIR}/*.frag" "${SHADER_DIR}/*.comp")

foreach(SHADER ${SHADERS})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
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