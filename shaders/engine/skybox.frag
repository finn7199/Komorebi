#version 450

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Skybox fragment shader — samples the environment cubemap and tone maps.
// The vertex shader provides a world-space direction per pixel.
// We normalize it and sample the cubemap, then apply Reinhard tone mapping
// so HDR values (sun, bright sky) don't clip to flat white.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layout(set = 1, binding = 0) uniform samplerCube envMap;

layout(location = 0) in vec3 worldDir;
layout(location = 0) out vec4 outColor;

void main() {
    // Sample the cubemap using the world direction from the vertex shader
    vec3 color = texture(envMap, normalize(worldDir)).rgb;

    // Reinhard tone mapping: compresses HDR range into 0-1.
    // value / (value + 1) — bright values stay bright but never clip.
    color = color / (color + vec3(1.0));

    // Gamma correction (linear → sRGB) for correct display
    color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}
