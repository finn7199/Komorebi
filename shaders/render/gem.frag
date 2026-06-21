#version 450
#include "kmrb_lighting.glsl"

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Gem fragment shader — faceted, iridescent polished crystal.
// Pairs with gem.vert (which spins the mesh for a seamless showcase loop).
//
// Two ideas drive the look:
//   1. FACETING — the screen-space derivative of world position is constant
//      across a triangle, so cross(dFdx, dFdy) is that face's flat normal.
//      Blending it toward the smooth mesh normal turns a smooth sphere into a
//      cut crystal without changing the geometry. Dial it with `facet`.
//   2. IRIDESCENCE — a cosine spectrum palette keyed to the view angle, so
//      every facet flashes a different hue as the gem rotates. Concentrated on
//      the rim by a Fresnel term, like thin-film on a polished stone.
//
// Inspector sliders (auto-discovered by SPIRV-Reflect):
//   iridescence — rainbow rim intensity (try 0.5–1.5)
//   hueShift    — rotate the spectrum (0–1) to retint the whole gem
//   facet       — 0 = smooth orb, 1 = hard crystal facets
//   roughness   — surface polish: 0 = mirror, ~0.15 = glassy gem
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

layout(push_constant) uniform PushConstants {
    mat4  model;        // offset  0 — engine built-in (must match gem.vert)
    vec4  color;        // offset 64 — base gem tint
    float iridescence;  // offset 80 — Inspector slider
    float hueShift;     // offset 84 — Inspector slider
    float facet;        // offset 88 — Inspector slider
    float roughness;    // offset 92 — Inspector slider
} push;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

// Inigo-Quilez-style cosine palette — cheap, vivid full spectrum over t in [0,1].
vec3 spectrum(float t) {
    return 0.5 + 0.5 * cos(6.28318530718 * (t + vec3(0.0, 0.33, 0.67)));
}

void main() {
    // ── Faceted normal ──
    // Flat per-face normal from world-position derivatives, blended toward the
    // interpolated smooth normal. facet=0 → orb, facet=1 → hard crystal.
    vec3 flatN   = normalize(cross(dFdx(fragWorldPos), dFdy(fragWorldPos)));
    vec3 smoothN = normalize(fragNormal);
    // The derivative normal's sign depends on screen-space winding / API, so orient it to
    // the geometric normal — it must never point inward, regardless of mesh winding.
    if (dot(flatN, smoothN) < 0.0) flatN = -flatN;
    vec3 N = normalize(mix(smoothN, flatN, clamp(push.facet, 0.0, 1.0)));

    vec3 V = normalize(global.cameraPos.xyz - fragWorldPos);

    // ── Polished gem core ──
    // Metallic + low roughness so it grabs crisp scene-light highlights and,
    // when an HDR is loaded, mirror-like IBL reflections in every facet.
    vec3 core = kmrb_pbr(fragWorldPos, N, V, push.color.rgb, 0.9, max(push.roughness, 0.03));

    // ── Iridescent rim ──
    // Hue rides the view angle (N·V), so it shifts as the gem spins — no time
    // term needed, which keeps the rotation loop perfectly seamless.
    float NdotV   = max(dot(N, V), 0.0);
    float fresnel = pow(1.0 - NdotV, 3.0);
    float hue     = fract(NdotV * 0.5 + push.hueShift);
    vec3  irid    = spectrum(hue) * fresnel * push.iridescence;

    // ── Combine + tone-map ──
    // Reinhard stays in LINEAR space (no pow(1/2.2)); the sRGB swapchain does
    // the gamma encode on write, matching mesh_pbr.frag's convention.
    vec3 result = core + irid;
    result = result / (result + vec3(1.0));

    outColor = vec4(result, push.color.a);
}
