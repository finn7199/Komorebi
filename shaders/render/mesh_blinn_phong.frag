#version 450

// Example: Blinn-Phong lit fragment shader for mesh entities.
// Reads light data from Global UBO and applies diffuse + specular shading.
// Users can modify this or write their own PBR/toon/custom shaders.

struct GPULight {
    vec4 positionAndType;       // xyz = position, w = type (0=point, 1=dir, 2=spot)
    vec4 directionAndAngle;     // xyz = direction, w = cos(spot angle)
    vec4 colorAndIntensity;     // xyz = color, w = intensity
    vec4 params;                // x = radius, yzw = reserved
};

layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
    float _pad0;      // Must be individual floats, NOT float[2] — std140 pads array elements to 16 bytes
    float _pad1;
    GPULight lights[8];
    int lightCount;
} global;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 color;
} push;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(global.cameraPos.xyz - fragWorldPos);

    // Ambient
    vec3 result = vec3(0.05) * push.color.rgb;

    for (int i = 0; i < global.lightCount; i++) {
        vec3 lightColor = global.lights[i].colorAndIntensity.xyz;
        float intensity = global.lights[i].colorAndIntensity.w;
        float lightType = global.lights[i].positionAndType.w;

        vec3 L;
        float attenuation = 1.0;

        if (lightType < 0.5) {
            // Point light
            vec3 toLight = global.lights[i].positionAndType.xyz - fragWorldPos;
            float dist = length(toLight);
            L = toLight / dist;
            float radius = global.lights[i].params.x;
            attenuation = clamp(1.0 - (dist * dist) / (radius * radius), 0.0, 1.0);
        } else {
            // Directional light
            L = normalize(-global.lights[i].directionAndAngle.xyz);
        }

        // Diffuse
        float diff = max(dot(N, L), 0.0);

        // Specular (Blinn-Phong)
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(N, H), 0.0), 32.0);

        result += (diff * push.color.rgb + spec * vec3(0.3)) * lightColor * intensity * attenuation;
    }

    outColor = vec4(result, push.color.a);
}
