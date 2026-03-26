#include "kmrb_sim.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace kmrb {

void Simulation::init(entt::registry& registry, uint32_t count) {
    particleCount = count;

    srand(42);
    for (uint32_t i = 0; i < count; i++) {
        auto entity = registry.create();

        // Random point in a sphere (radius 1.5)
        float theta = static_cast<float>(rand()) / RAND_MAX * 6.2831853f;
        float phi = acos(1.0f - 2.0f * static_cast<float>(rand()) / RAND_MAX);
        float r = cbrt(static_cast<float>(rand()) / RAND_MAX) * 1.5f;

        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        registry.emplace<ParticleTag>(entity);
        registry.emplace<Position>(entity, glm::vec3(x, y, z));
        registry.emplace<Velocity>(entity, glm::vec3(0.0f));
        registry.emplace<Mass>(entity, 1.0f);
        registry.emplace<PointSize>(entity, 2.0f);
        registry.emplace<Color>(entity, glm::vec4(
            0.4f + 0.6f * (r / 1.5f),
            0.2f + 0.3f * (1.0f - r / 1.5f),
            0.8f, 1.0f
        ));
    }

    std::cout << "[KMRB] ECS initialized (" << count << " particle entities)" << std::endl;
}

// Flatten ECS components into GPU-ready Particle structs
std::vector<Particle> Simulation::syncToSSBO(entt::registry& registry) {
    std::vector<Particle> particles;
    particles.reserve(particleCount);

    // EnTT view iterates only entities that have ALL listed components
    auto view = registry.view<ParticleTag, Position, Velocity, PointSize, Color>();

    view.each([&](auto entity, auto& pos, auto& vel, auto& size, auto& color) {
        Particle p{};
        p.position = glm::vec4(pos.pos, size.size);
        p.velocity = glm::vec4(vel.vel, 0.0f);
        p.color = color.rgba;
        particles.push_back(p);
        });

    return particles;
}

} // namespace kmrb
