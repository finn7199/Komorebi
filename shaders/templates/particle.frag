#version 450

// ── Environment Map (optional, set in Scene > Environment) ──
// Cubemap sampler bound by the engine. Returns black if no HDR is loaded.
// Sample with a direction vector: texture(envMap, direction).rgb
//
// Common uses:
//   Reflection:  texture(envMap, reflect(-viewDir, normal)).rgb
//   Ambient:     texture(envMap, normal).rgb * 0.2
//   Sky color:   texture(envMap, vec3(0, 1, 0)).rgb
//
// Values are HDR (can be > 1.0). Tone map if outputting directly:
//   color = color / (color + vec3(1.0));  // Reinhard
//
layout(set = 1, binding = 0) uniform samplerCube envMap;

// ── Inputs from vertex shader ──
layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    // Round point sprite (discard pixels outside circle)
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (dot(coord, coord) > 0.25) discard;

    // Example: sample environment map upward for ambient tint
    // vec3 ambient = texture(envMap, vec3(0, 1, 0)).rgb * 0.1;

    outColor = vec4(fragColor, 1.0);
}
