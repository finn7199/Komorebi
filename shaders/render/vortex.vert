#version 450

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Vortex vertex shader — reads SSBO particles, outputs billboard points.
// Each point becomes a camera-facing quad via gl_PointSize + gl_PointCoord.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

struct Particle {
    vec4 position;  // xyz = pos, w = point size
    vec4 velocity;  // xyz = vel, w = lifetime
    vec4 color;     // rgba
};

// Binding 1 = output buffer (the one compute just wrote to this frame)
layout(set = 2, binding = 1) readonly buffer ParticleSSBO {
    Particle particles[];
} ssbo;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragSpeed;

void main() {
    Particle p = ssbo.particles[gl_VertexIndex];

    vec4 worldPos = push.model * vec4(p.position.xyz, 1.0);
    gl_Position = global.proj * global.view * worldPos;

    // Billboard size — particles closer to center are smaller and brighter
    gl_PointSize = max(p.position.w, 1.5);

    // Pass speed to fragment shader for color variation
    fragSpeed = length(p.velocity.xyz);
    fragColor = p.color.rgb * push.color.rgb;
}
