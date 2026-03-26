#version 450

// Set 0: Global — camera, time
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

// Push constants — per-draw transform
layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

// Set 2: Double-buffered particle SSBOs
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

void main() {
    Particle p = ssbo.particles[gl_VertexIndex];

    vec4 worldPos = push.model * vec4(p.position.xyz, 1.0);
    gl_Position = global.proj * global.view * worldPos;
    gl_PointSize = p.position.w;
    fragColor = p.color.rgb * push.color.rgb;
}
