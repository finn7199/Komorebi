#pragma once

#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <vector>

#include "kmrb_types.hpp"

namespace kmrb {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ECS COMPONENTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Small, data-only structs. EnTT stores them contiguously in memory.

struct Position {
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

// Tag component — marks entities as particles (zero-size, just for filtering)
struct ParticleTag {};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SIMULATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Simulation {
public:
    // Create particle entities in the registry
    void init(entt::registry& registry, uint32_t count);

    // Gather ECS component data into a flat Particle array for GPU upload
    // Call this at init and whenever you need to push CPU state to the SSBO
    std::vector<Particle> syncToSSBO(entt::registry& registry);

    uint32_t getParticleCount() const { return particleCount; }

private:
    uint32_t particleCount = 0;
};

} // namespace kmrb
