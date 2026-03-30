#version 450

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

layout(location = 0) in vec3 inPosition;

layout(location = 0) out vec4 fragColor;

void main() {
    gl_Position = global.proj * global.view * push.model * vec4(inPosition, 1.0);
    fragColor = push.color;
}
