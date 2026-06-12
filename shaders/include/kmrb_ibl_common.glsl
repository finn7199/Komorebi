#ifndef KMRB_IBL_COMMON_GLSL
#define KMRB_IBL_COMMON_GLSL

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Shared math for the IBL precompute shaders (brdf_lut / irradiance / prefilter).
//
// IBL (image-based lighting) treats the environment map as a light source.
// The full lighting integral is too expensive per-pixel, so the engine bakes
// it into lookup textures using the "split-sum" approximation (Karis, UE4):
//
//   ∫ Li(l)·BRDF(l,v)·cosθ dl  ≈  prefiltered(R, roughness) × brdfLUT(NdotV, roughness)
//
// This file holds the building blocks: low-discrepancy sampling (Hammersley),
// GGX importance sampling, and the cubemap texel → world direction mapping.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const float KMRB_IBL_PI = 3.14159265359;

// Map a cubemap texel (x, y, face) to its world-space direction.
// MUST match the CPU-side equirect→cube conversion in kmrb_renderer.cpp
// (loadEnvironmentMap) — same face order and orientation, or the baked maps
// would be rotated relative to the skybox.
// face 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z
vec3 kmrb_cubeDir(ivec3 texel, int faceSize) {
    // Texel center → [-1, 1] on the face plane
    float u = (float(texel.x) + 0.5) / float(faceSize) * 2.0 - 1.0;
    float v = (float(texel.y) + 0.5) / float(faceSize) * 2.0 - 1.0;

    vec3 dir;
    switch (texel.z) {
        case 0: dir = vec3( 1.0,  -v,  -u); break; // +X
        case 1: dir = vec3(-1.0,  -v,   u); break; // -X
        case 2: dir = vec3(   u, 1.0,   v); break; // +Y
        case 3: dir = vec3(   u,-1.0,  -v); break; // -Y
        case 4: dir = vec3(   u,  -v, 1.0); break; // +Z
        default:dir = vec3(  -u,  -v,-1.0); break; // -Z
    }
    return normalize(dir);
}

// Van der Corput radical inverse — bit-reverses i as a fraction in [0,1).
// Together with i/N this forms the Hammersley sequence: quasi-random points
// that cover the unit square far more evenly than pseudo-random numbers,
// so the integral converges with fewer samples.
float kmrb_radicalInverse_VdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}

vec2 kmrb_hammersley(uint i, uint N) {
    return vec2(float(i) / float(N), kmrb_radicalInverse_VdC(i));
}

// Turn a uniform 2D sample Xi into a half-vector H distributed like the GGX
// specular lobe around N. "Importance sampling": we spend samples where the
// BRDF is large instead of uniformly over the hemisphere, which is the only
// way 1024 samples can approximate an integral over millions of directions.
vec3 kmrb_importanceSampleGGX(vec2 Xi, vec3 N, float roughness) {
    float a = roughness * roughness;

    // GGX inverse transform: maps Xi.y to a polar angle with GGX's density
    float phi      = 2.0 * KMRB_IBL_PI * Xi.x;
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a * a - 1.0) * Xi.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    // Spherical → cartesian, in tangent space (Z = N)
    vec3 H = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Tangent space → world space around N
    vec3 up      = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
    return normalize(tangent * H.x + bitangent * H.y + N * H.z);
}

// Smith geometry term for IBL. Note k = a²/2 here, NOT the (r+1)²/8 used for
// analytic lights in kmrb_lighting.glsl — the two variants are calibrated for
// their respective integration domains (Karis 2013).
float kmrb_G_SmithIBL(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float k = (a * a) / 2.0;
    float gv = NdotV / (NdotV * (1.0 - k) + k);
    float gl = NdotL / (NdotL * (1.0 - k) + k);
    return gv * gl;
}

#endif // KMRB_IBL_COMMON_GLSL
