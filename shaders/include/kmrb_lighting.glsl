#ifndef KMRB_LIGHTING_GLSL
#define KMRB_LIGHTING_GLSL

// Shared lighting library — #include "kmrb_lighting.glsl" in any fragment shader.
//
// Light types match the dropdown in the UI and the LightType enum in kmrb_sim.hpp:
//   Point       = 0  — position + radius, fades to zero at the radius boundary
//   Directional = 1  — sun-like, infinite distance, no attenuation
//   Spot        = 2  — position + direction + cone angle, soft penumbra at the edge
//
// The engine packs all scene LightComponents into GlobalUBO every frame,
// so adding/removing a light entity immediately affects every shader that uses this file.

struct GPULight {
    vec4 positionAndType;    // xyz = world position,   w = type (0/1/2)
    vec4 directionAndAngle;  // xyz = aim direction,    w = cos(spot half-angle)
    vec4 colorAndIntensity;  // xyz = RGB color,        w = intensity
    vec4 params;             // x   = radius (point/spot), yzw = reserved
};

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
    float _pad0;
    float _pad1;
    GPULight lights[8];
    int lightCount;
} global;

// Resolved per-light data — output of kmrb_resolveLight().
struct KmrbLight {
    vec3  L;     // direction from the fragment toward the light (normalized)
    float atten; // combined distance + cone attenuation [0..1]
    vec3  color; // light color * intensity (pre-multiplied, ready to use)
};

// Resolve light i for a fragment at worldPos.
// Reads the type from positionAndType.w and runs the matching branch.
// Use this in custom shaders (thinfilm, etc.) to get per-light L/atten/color
// without reimplementing the point/directional/spot branching yourself.
KmrbLight kmrb_resolveLight(int i, vec3 worldPos) {
    KmrbLight r;
    r.color = global.lights[i].colorAndIntensity.xyz * global.lights[i].colorAndIntensity.w;
    r.atten = 1.0;
    r.L     = vec3(0.0, 1.0, 0.0); // safe fallback

    float lightType = global.lights[i].positionAndType.w;

    if (lightType < 0.5) {
        // ── Point light ──
        vec3  toLight = global.lights[i].positionAndType.xyz - worldPos;
        float dist    = length(toLight);
        r.L           = (dist > 0.001) ? toLight / dist : vec3(0.0, 1.0, 0.0);
        float radius  = global.lights[i].params.x;
        r.atten       = clamp(1.0 - (dist * dist) / (radius * radius), 0.0, 1.0);

    } else if (lightType < 1.5) {
        // ── Directional light ──
        // directionAndAngle.xyz is the direction the light is pointing (toward the scene).
        // Negate it to get L (fragment → light source).
        r.L = normalize(-global.lights[i].directionAndAngle.xyz);

    } else {
        // ── Spot light ──
        // Position-based like point, plus a cone that fades at the edges.
        vec3  toLight  = global.lights[i].positionAndType.xyz - worldPos;
        float dist     = length(toLight);
        r.L            = (dist > 0.001) ? toLight / dist : vec3(0.0, 1.0, 0.0);

        float radius    = global.lights[i].params.x;
        float distAtten = clamp(1.0 - (dist * dist) / (radius * radius), 0.0, 1.0);

        // Cone check: dot(-L, spotDir) gives the cosine of the angle between
        // the fragment direction and where the spot is aimed.
        // cosOuter is cos(spotAngle) set in the UI — fragment outside the cone gets 0.
        // cosInner is 80% inside the cone — smoothstep gives a soft penumbra at the edge.
        vec3  spotDir   = normalize(global.lights[i].directionAndAngle.xyz);
        float cosOuter  = global.lights[i].directionAndAngle.w;
        float cosInner  = mix(1.0, cosOuter, 0.8);
        float theta     = dot(-r.L, spotDir);
        float coneAtten = smoothstep(cosOuter, cosInner, theta);

        r.atten = distAtten * coneAtten;
    }

    return r;
}

// ── PBR helpers ──────────────────────────────────────────────────────────────

const float KMRB_PI = 3.14159265359;

// GGX Normal Distribution Function — how many microfacets point toward H
float kmrb_D_GGX(float NdotH, float roughness) {
    float a  = roughness * roughness;
    float a2 = a * a;
    float d  = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (KMRB_PI * d * d);
}

// Smith-Schlick Geometry — how much microfacets block each other (view + light sides)
float kmrb_G_Smith(float NdotV, float NdotL, float roughness) {
    float k  = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

// Schlick Fresnel — more reflection at grazing angles
vec3 kmrb_F_Schlick(float HdotV, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - HdotV, 0.0, 1.0), 5.0);
}

// Cook-Torrance PBR shading summed over all scene lights.
//   worldPos  — fragment position in world space
//   N         — surface normal, normalized
//   V         — view direction (cameraPos - worldPos), normalized
//   albedo    — base color (push.color.rgb)
//   metallic  — 0 = plastic/wood, 1 = gold/iron  (Inspector slider)
//   roughness — 0 = mirror smooth, 1 = fully rough (Inspector slider)
// Returns the final lit RGB with a small image-based ambient term.
vec3 kmrb_pbr(vec3 worldPos, vec3 N, vec3 V, vec3 albedo, float metallic, float roughness) {
    roughness = max(roughness, 0.05); // clamp: roughness=0 makes GGX D-term collapse to 0 → black surface

    // F0 = base reflectivity at normal incidence.
    // Dielectrics (plastic, skin) use 0.04 (4% reflection straight-on).
    // Metals use the albedo itself as their reflectivity tint.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    vec3  Lo    = vec3(0.0);
    float NdotV = max(dot(N, V), 0.001);

    for (int i = 0; i < global.lightCount; i++) {
        KmrbLight light = kmrb_resolveLight(i, worldPos);
        vec3  H     = normalize(V + light.L);
        float NdotL = max(dot(N, light.L), 0.0);
        float NdotH = max(dot(N, H),       0.0);
        float HdotV = max(dot(H, V),       0.0);

        // Cook-Torrance specular BRDF
        float D = kmrb_D_GGX(NdotH, roughness);
        float G = kmrb_G_Smith(NdotV, NdotL, roughness);
        vec3  F = kmrb_F_Schlick(HdotV, F0);
        vec3  specular = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

        // Diffuse — metals have no diffuse (energy absorbed into free electrons)
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        Lo += (kD * albedo / KMRB_PI + specular) * light.color * light.atten * NdotL;
    }

    vec3 ambient = vec3(0.03) * albedo; // simple flat ambient; replace with IBL for full PBR
    return ambient + Lo;
}

#endif // KMRB_LIGHTING_GLSL
