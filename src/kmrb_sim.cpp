#include "kmrb_ui.hpp"
#include "kmrb_sim.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>

namespace kmrb {

void Simulation::init(entt::registry& registry, uint32_t count) {
    particleCount = count;

    // Create zeroed particle entities — the init shader sets actual positions on the GPU.
    for (uint32_t i = 0; i < count; i++) {
        auto entity = registry.create();
        registry.emplace<ParticleTag>(entity);
        registry.emplace<ParticlePosition>(entity, glm::vec3(0.0f));
        registry.emplace<Velocity>(entity, glm::vec3(0.0f));
        registry.emplace<Mass>(entity, 1.0f);
        registry.emplace<PointSize>(entity, 2.0f);
        registry.emplace<Color>(entity, glm::vec4(1.0f));
    }

    kmrb::Log::ok("ECS initialized (" + std::to_string(count) + " particle entities)");
}

// Flatten ECS components into GPU-ready Particle structs
std::vector<Particle> Simulation::syncToSSBO(entt::registry& registry) {
    std::vector<Particle> particles;
    particles.reserve(particleCount);

    // EnTT view iterates only entities that have ALL listed components
    auto view = registry.view<ParticleTag, ParticlePosition, Velocity, PointSize, Color>();

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
