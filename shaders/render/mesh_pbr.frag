#version 450
#include "kmrb_lighting.glsl"

// PBR mesh shader (Cook-Torrance: GGX + Smith + Schlick Fresnel).
// All three light types (point / directional / spot) handled automatically.
//
// Inspector sliders (auto-discovered by SPIRV-Reflect):
//   metallic  — 0 = plastic / wood / skin,  1 = gold / iron / copper
//   roughness — 0 = mirror smooth,           1 = fully rough / matte

layout(push_constant) uniform PushConstants {
    mat4  model;      // offset  0 — engine built-in
    vec4  color;      // offset 64 — base albedo (rgb) + alpha
    float metallic;   // offset 80 — Inspector slider
    float roughness;  // offset 84 — Inspector slider
} push;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(global.cameraPos.xyz - fragWorldPos);
    vec3 lit = kmrb_pbr(fragWorldPos, N, V, push.color.rgb, push.metallic, push.roughness);
    outColor = vec4(lit, push.color.a);
}
