#version 450

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Gem vertex shader — spins a mesh in-shader for a seamless showcase loop.
//
// Drop-in compatible with mesh fragment shaders (thinfilm.frag, mesh_pbr.frag):
// it outputs the same varyings (worldPos / normal / uv) as mesh_basic.vert.
//
// Why spin here instead of rotating the entity transform?
//   The rotation is driven by global.time and a fixed PERIOD, so the motion
//   is perfectly periodic. Record exactly ROTATION_PERIOD seconds and the GIF
//   loops with no visible seam — the last frame matches the first.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Set 0: Global — camera and scene data
layout(set = 0, binding = 0) uniform GlobalUBO {
    mat4 view;
    mat4 proj;
    vec4 cameraPos;
    float time;
    float deltaTime;
} global;

// Per-draw push constants. This block is a prefix of thinfilm.frag's block
// (model + color), so the two stages share one pipeline layout cleanly.
layout(push_constant) uniform PushConstants {
    mat4 model;   // offset  0 — engine built-in transform
    vec4 color;   // offset 64 — base tint
} push;

// Mesh vertex attributes (from the vertex buffer)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

// Pass to fragment shader (must match thinfilm.frag's inputs)
layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;

// ── Tunables ──
const float PI              = 3.14159265359;
const float ROTATION_PERIOD = 8.0;   // seconds for one full turn — record this long for a clean loop
const float TILT            = 0.35;   // constant lean (radians) so we see the top, not a dead-on spin

// Build a rotation matrix about an arbitrary axis (Rodrigues' formula).
mat3 rotateAxis(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float t = 1.0 - c;
    return mat3(
        t*axis.x*axis.x + c,        t*axis.x*axis.y - s*axis.z, t*axis.x*axis.z + s*axis.y,
        t*axis.x*axis.y + s*axis.z, t*axis.y*axis.y + c,        t*axis.y*axis.z - s*axis.x,
        t*axis.x*axis.z - s*axis.y, t*axis.y*axis.z + s*axis.x, t*axis.z*axis.z + c
    );
}

void main() {
    // Continuous yaw that completes exactly one turn per ROTATION_PERIOD.
    float angle = global.time * (2.0 * PI / ROTATION_PERIOD);

    // Spin about Y, then lean the whole thing back a touch so the silhouette
    // shows depth instead of a flat profile.
    mat3 spin = rotateAxis(vec3(0.0, 1.0, 0.0), angle);
    mat3 lean = rotateAxis(vec3(1.0, 0.0, 0.0), TILT);
    mat3 rot  = lean * spin;

    vec3 spunPos    = rot * inPosition;
    vec3 spunNormal = rot * inNormal;

    vec4 worldPos = push.model * vec4(spunPos, 1.0);
    gl_Position   = global.proj * global.view * worldPos;

    fragWorldPos = worldPos.xyz;
    // Rotation is orthonormal, so the normal can use the same matrix as position
    // (no inverse-transpose needed) before the model matrix's normal transform.
    fragNormal   = mat3(transpose(inverse(push.model))) * spunNormal;
    fragUV       = inUV;
}
