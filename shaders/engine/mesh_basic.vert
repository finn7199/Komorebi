#version 450

// Global UBO — camera and scene data
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

// Per-draw push constants
layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

// Mesh vertex attributes (from vertex buffer)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// Pass to fragment shader
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    gl_Position   = global.proj * global.view * worldPos;

    fragWorldPos = worldPos.xyz;
    fragNormal   = mat3(transpose(inverse(push.model))) * inNormal;
    fragUV       = inUV;
}
