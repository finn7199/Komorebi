#version 450

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Vortex fragment shader — soft radial glow billboard.
// gl_PointCoord gives us a 0-1 UV across the point sprite quad.
// Instead of a hard circle cutoff, this creates a smooth falloff
// with a bright center that fades to transparent edges.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layout(location = 0) in vec3 fragColor;
layout(location = 1) in float fragSpeed;

layout(location = 0) out vec4 outColor;

void main() {
    // Distance from center of the point sprite (0 = center, 0.5 = edge)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);

    // Discard pixels outside the circle
    if (dist > 0.5) discard;

    // Soft glow falloff — bright center, fading edges
    float glow = 1.0 - smoothstep(0.0, 0.5, dist);
    glow = pow(glow, 1.5);  // Sharpen the falloff slightly

    // Tint faster particles toward white/hot, slower toward base color
    float heat = clamp(fragSpeed * 0.3, 0.0, 1.0);
    vec3 hotColor = mix(fragColor, vec3(1.0, 0.95, 0.8), heat);

    outColor = vec4(hotColor * glow, glow);
}
