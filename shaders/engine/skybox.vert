#version 450

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Skybox vertex shader — fullscreen triangle trick.
// Renders a single triangle that covers the entire screen (3 vertices, no VBO).
// Unprojects each pixel from clip space back to a world-space direction,
// which the fragment shader uses to sample the environment cubemap.
// Depth is set to 1.0 (far plane) so everything else draws on top.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

layout(location = 0) out vec3 worldDir;

void main() {
    // Generate fullscreen triangle vertices from gl_VertexIndex (0, 1, 2).
    // This produces a triangle that covers the entire viewport without a vertex buffer.
    vec2 uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec4 clipPos = vec4(uv * 2.0 - 1.0, 1.0, 1.0);

    // Remove translation from the view matrix so the skybox stays centered on the camera.
    // Only rotation matters — the sky is infinitely far away.
    mat4 viewNoTranslation = mat4(mat3(global.view));
    mat4 invViewProj = inverse(global.proj * viewNoTranslation);

    // Unproject from clip space to world direction
    vec4 worldPos = invViewProj * clipPos;
    worldDir = worldPos.xyz / worldPos.w;

    // Output position at z=1.0 (far plane) so skybox is behind everything
    gl_Position = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
}
