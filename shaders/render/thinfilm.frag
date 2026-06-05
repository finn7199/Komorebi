#version 450
#include "kmrb_lighting.glsl"

// Thin-film interference shader — soap bubble / iridescent coating effect.
// Uses kmrb_resolveLight() from the include to get per-light direction and attenuation,
// then runs the full thin-film optics (Fresnel + phase interference) per wavelength.
// All three light types work automatically — just add lights to the scene.

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
    float filmThickness; // Base coating thickness (nm), e.g. 200–800
    float thicknessWave; // Wavy noise animation intensity, e.g. 0–200
    float n_film;        // Refractive index of the film layer, e.g. 1.1–2.0
    float n_base;        // Refractive index of the base material, e.g. 1.3–2.5
} push;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

const float PI    = 3.14159265359;
const float n_air = 1.0;

// ── Complex number helpers ──
struct Complex { float real; float imag; };
Complex c_add(Complex a, Complex b) { return Complex(a.real + b.real, a.imag + b.imag); }
Complex c_mul(Complex a, Complex b) { return Complex(a.real*b.real - a.imag*b.imag, a.real*b.imag + a.imag*b.real); }
Complex c_div(Complex a, Complex b) {
    float d = b.real*b.real + b.imag*b.imag;
    return Complex((a.real*b.real + a.imag*b.imag) / d, (a.imag*b.real - a.real*b.imag) / d);
}
float c_abs_sq(Complex c) { return c.real*c.real + c.imag*c.imag; }

// ── Thin-film reflectance for one wavelength ──
// cosTheta1 = cos of incidence angle, lambda = wavelength in nm
// r12/r23 = Fresnel coefficients at the two interfaces
float computeReflectance(float cosTheta1, float lambda, float r12, float r23,
                         float thickness, float nf, float nb) {
    float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
    float sinTheta2Sq = (n_air * n_air) / (nf * nf) * sinTheta1Sq;
    if (sinTheta2Sq > 1.0) return 1.0;
    float cosTheta2 = sqrt(1.0 - sinTheta2Sq);

    float phi   = 2.0 * thickness * nf * cosTheta2;
    float delta = (2.0 * PI * phi) / lambda;

    Complex e_id  = Complex(cos(delta), sin(delta));
    Complex num   = c_add(Complex(r12, 0.0), c_mul(Complex(r23, 0.0), e_id));
    Complex denom = c_add(Complex(1.0, 0.0), c_mul(Complex(r12*r23, 0.0), e_id));
    return clamp(c_abs_sq(c_div(num, denom)), 0.0, 1.0);
}

// Procedural film thickness variation driven by thicknessWave slider
float getDynamicThickness(vec2 uv) {
    float noise = sin(uv.x * 35.0 + global.time * 2.0) * cos(uv.y * 35.0 + global.time * 1.5)
                + sin((uv.x + uv.y) * 20.0 - global.time);
    return push.filmThickness + noise * push.thicknessWave;
}

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(global.cameraPos.xyz - fragWorldPos);

    float nf               = push.n_film;
    float nb               = push.n_base;
    float currentThickness = getDynamicThickness(fragUV);
    vec3  baseColor        = push.color.rgb * 0.01; // Dark base so iridescence pops
    vec3  result           = vec3(0.0);

    for (int i = 0; i < global.lightCount; i++) {
        // kmrb_resolveLight handles point / directional / spot automatically.
        KmrbLight light = kmrb_resolveLight(i, fragWorldPos);

        float diff = max(dot(N, light.L), 0.0);
        vec3  H    = normalize(light.L + V);

        // Use N·H as the incidence angle for thin-film optics.
        // Fall back to N·V when the half-vector is near zero (grazing view).
        float cosTheta1 = clamp(dot(N, H), 0.0, 1.0);
        if (cosTheta1 < 0.001) cosTheta1 = clamp(dot(N, V), 0.1, 1.0);

        // ── Fresnel coefficients at both interfaces ──
        float sinTheta1Sq = 1.0 - cosTheta1 * cosTheta1;
        float cosTheta2   = sqrt(clamp(1.0 - (n_air*n_air)/(nf*nf) * sinTheta1Sq, 0.0, 1.0));
        float cosTheta3   = sqrt(clamp(1.0 - (nf*nf)/(nb*nb) * (1.0 - cosTheta2*cosTheta2), 0.0, 1.0));

        float r12_s = (n_air*cosTheta1 - nf*cosTheta2) / (n_air*cosTheta1 + nf*cosTheta2);
        float r23_s = (nf*cosTheta2   - nb*cosTheta3)  / (nf*cosTheta2   + nb*cosTheta3);
        float r12_p = (nf*cosTheta1   - n_air*cosTheta2) / (nf*cosTheta1 + n_air*cosTheta2);
        float r23_p = (nb*cosTheta2   - nf*cosTheta3)    / (nb*cosTheta2 + nf*cosTheta3);

        // ── Per-wavelength reflectance (R, G, B) ──
        vec3 wavelengths = vec3(650.0, 530.0, 460.0);
        vec3 R_s = vec3(
            computeReflectance(cosTheta1, wavelengths.r, r12_s, r23_s, currentThickness, nf, nb),
            computeReflectance(cosTheta1, wavelengths.g, r12_s, r23_s, currentThickness, nf, nb),
            computeReflectance(cosTheta1, wavelengths.b, r12_s, r23_s, currentThickness, nf, nb)
        );
        vec3 R_p = vec3(
            computeReflectance(cosTheta1, wavelengths.r, r12_p, r23_p, currentThickness, nf, nb),
            computeReflectance(cosTheta1, wavelengths.g, r12_p, r23_p, currentThickness, nf, nb),
            computeReflectance(cosTheta1, wavelengths.b, r12_p, r23_p, currentThickness, nf, nb)
        );
        vec3 interference = 0.5 * (R_s + R_p);

        float specMask = pow(max(dot(N, H), 0.0), 2.0);
        result += (diff * baseColor + specMask * interference * 15.0) * light.color * light.atten;
    }

    result   = pow(result, vec3(1.0 / 2.2)); // gamma correct
    outColor = vec4(result, push.color.a);
}
