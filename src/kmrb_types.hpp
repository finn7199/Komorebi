#pragma once

#include <vulkan/vulkan.hpp>
#include <glm/glm.hpp>
#include <array>

namespace kmrb {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESCRIPTOR SET CONVENTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//   Set 0: Global     — camera, time, environment (V1 uses this)
//   Set 1: Material   — textures, PBR params      (V2)
//   Set 2: Per-object — model transform, instance  (V2)
enum DescriptorSet : uint32_t {
    DESCRIPTOR_SET_GLOBAL   = 0,
    DESCRIPTOR_SET_MATERIAL = 1,
    DESCRIPTOR_SET_OBJECT   = 2,
    DESCRIPTOR_SET_COUNT    = 3
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GLOBAL UBO (Set 0, Binding 0)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
struct GlobalUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 cameraPos;    // xyz = position, w = unused
    float time;
    float deltaTime;
    float padding[2];       // Align to 16 bytes (std140 layout rules)
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PUSH CONSTANTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 128 bytes guaranteed by Vulkan spec. Used for per-draw data that changes often.
// V1: model matrix for particles. V2: model matrix + material index.
struct PushConstants {
    glm::mat4 model;        // 64 bytes
    glm::vec4 color;        // 16 bytes — tint/debug color
    // 48 bytes remaining for V2 (material ID, flags, etc.)
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PARTICLE (SSBO layout — Set 2, Binding 0)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// This struct is shared between CPU (upload) and GPU (shader).
// Uses vec4 for alignment — GLSL std430 requires 16-byte alignment for vec4.
// Compute shaders will write to this, vertex shaders read it.
struct Particle {
    glm::vec4 position;  // xyz = position, w = point size
    glm::vec4 velocity;  // xyz = velocity, w = lifetime
    glm::vec4 color;     // rgba
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VERTEX FORMATS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// V1: position + color. V2: position + normal + uv + tangent.
// Configurable — pipeline takes binding/attribute descriptions
struct Vertex {
    glm::vec3 position;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
        return { 0, sizeof(Vertex), vk::VertexInputRate::eVertex };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        return {{
            { 0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, position) },
            { 1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color) }
        }};
    }
};

// V2 will add:
// struct MeshVertex {
//     glm::vec3 position;
//     glm::vec3 normal;
//     glm::vec2 uv;
//     glm::vec4 tangent;
//     static binding/attribute descriptions...
// };

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RENDER PASS CONFIG
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Parameterized render pass creation — not hardcoded so V2 can add passes.
struct RenderPassConfig {
    vk::Format colorFormat;
    vk::AttachmentLoadOp loadOp       = vk::AttachmentLoadOp::eClear;
    vk::AttachmentStoreOp storeOp     = vk::AttachmentStoreOp::eStore;
    vk::ImageLayout initialLayout     = vk::ImageLayout::eUndefined;
    vk::ImageLayout finalLayout       = vk::ImageLayout::ePresentSrcKHR;
    vk::SampleCountFlagBits samples   = vk::SampleCountFlagBits::e1;
    bool hasDepth                     = false;       // V2: shadow/depth passes
    vk::Format depthFormat            = vk::Format::eD32Sfloat;
};

} // namespace kmrb
