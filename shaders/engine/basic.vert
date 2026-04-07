#version 450

// Set 0: Global UBO - camera, time, environment
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

// Push constants — per-draw data (model matrix, tint color)
layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

// Hardcoded triangle — replace by vertex buffer input
vec2 positions[3] = vec2[](
    vec2( 0.0,  1.0),   // Top center
    vec2(-1.0, -1.0),   // Bottom left
    vec2( 1.0, -1.0)    // Bottom right
);

vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 fragColor;

void main() {
    vec4 worldPos = push.model * vec4(positions[gl_VertexIndex], 0.0, 1.0);
    gl_Position = global.proj * global.view * worldPos;
    fragColor = colors[gl_VertexIndex] * push.color.rgb;
}
