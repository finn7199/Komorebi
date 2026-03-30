#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <vector>

#include "kmrb_types.hpp"

namespace kmrb {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UNIVERSAL COMPONENTS (every entity can have these)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

struct Name {
    std::string value;
};

// Every entity in the scene has a Transform
struct Transform {
    glm::vec3 position = { 0.0f, 0.0f, 0.0f };
    glm::vec3 rotation = { 0.0f, 0.0f, 0.0f }; // Euler degrees: pitch, yaw, roll
    glm::vec3 scale    = { 1.0f, 1.0f, 1.0f };
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OPTIONAL COMPONENTS (attached to define what an entity does)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Camera — viewpoint into the scene
struct CameraComponent {
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;
    bool active = false;
};

// Pipeline — SSBO + compute shader driven simulation
struct PipelineComponent {
    uint32_t particleCount = 10000;
};

// Shader program — attached to any entity that runs GPU programs
// Compute + vertex + fragment slots. Set dirty=true to trigger pipeline rebuild.
struct ShaderProgramComponent {
    std::string computePath;     // e.g., "shaders/compute/gravity.comp"
    std::string vertexPath;      // e.g., "shaders/render/particle.vert"
    std::string fragmentPath;    // e.g., "shaders/render/particle.frag"
    bool dirty = true;           // Set by UI/hot-reload, cleared after pipeline rebuild
};

// Grid helper — floor reference grid
struct GridComponent {
    float size = 10.0f;
    int cellCount = 10;
    glm::vec4 color = { 0.24f, 0.21f, 0.16f, 1.0f };
};

// Mesh renderer — V1: wireframe primitives. V2: full PBR meshes.
enum class PrimitiveShape { Cube, Sphere, Plane };
struct MeshRendererComponent {
    PrimitiveShape shape = PrimitiveShape::Cube;
    glm::vec4 color = { 0.6f, 0.6f, 0.6f, 1.0f };
    bool wireframe = true;
    // V2: materialID, mesh asset path, etc.
};

// Light — stub for V2 lighting
enum class LightType { Point, Directional, Spot };
struct LightComponent {
    LightType type = LightType::Point;
    glm::vec3 color = { 1.0f, 1.0f, 1.0f };
    float intensity = 1.0f;
    float radius = 10.0f;     // Point/spot
    float spotAngle = 45.0f;  // Spot only
};


// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PARTICLE DATA (used by Simulation::syncToSSBO, kept separate from ECS scene)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Tag for individual particle entities (the 10k entities, not the system entity)
struct ParticleTag {};

// Lightweight position for particles (10k+ entities — don't need full Transform)
struct ParticlePosition {
    glm::vec3 pos;
};

struct Velocity {
    glm::vec3 vel;
};

struct Mass {
    float value = 1.0f;
};

struct PointSize {
    float size = 2.0f;
};

struct Color {
    glm::vec4 rgba = { 1.0f, 1.0f, 1.0f, 1.0f };
};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SIMULATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Simulation {
public:
    void init(entt::registry& registry, uint32_t count);
    std::vector<Particle> syncToSSBO(entt::registry& registry);
    uint32_t getParticleCount() const { return particleCount; }

private:
    uint32_t particleCount = 0;
};

} // namespace kmrb
