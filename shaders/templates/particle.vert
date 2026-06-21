#version 450

// ── Global UBO (read-only, provided by engine) ──
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

// ── Push Constants ──
// Must match the compute shader's push constant layout.
layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

// ── Particle SSBO ──
// Binding 1 = the output buffer that compute just wrote to.
struct Particle {
    vec4 position;  // xyz = pos, w = point size
    vec4 velocity;  // xyz = vel, w = lifetime
    vec4 color;     // rgba
};

layout(set = 2, binding = 1) readonly buffer ParticleSSBO {
    Particle particles[];
} ssbo;

// ── Outputs to fragment shader ──
layout(location = 0) out vec3 fragColor;

void main() {
    Particle p = ssbo.particles[gl_VertexIndex];

    vec4 worldPos = push.model * vec4(p.position.xyz, 1.0);
    gl_Position = global.proj * global.view * worldPos;
    gl_PointSize = max(p.position.w, 1.0);

    fragColor = p.color.rgb * push.color.rgb;
}
