#include "kmrb_log.hpp"
#include "kmrb_sim.hpp"

namespace kmrb {

void Simulation::init(uint32_t count) {
    particleCount = count;
    kmrb::Log::ok("Simulation initialized (" + std::to_string(count) + " particles, GPU-resident)");
}

// Zeroed placeholder data for the first SSBO upload — the init compute shader
// writes the real positions/velocities on the GPU.
std::vector<Particle> Simulation::makeInitialSSBOData() const {
    return std::vector<Particle>(particleCount);  // Value-initialized = all zeros
}

} // namespace kmrb
