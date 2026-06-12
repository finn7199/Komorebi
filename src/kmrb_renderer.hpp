#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include <chrono>

#include "kmrb_types.hpp"
#include "kmrb_buffers.hpp"
#include "kmrb_mesh.hpp"
#include "kmrb_ui.hpp"
#include "kmrb_camera.hpp"
#include "kmrb_sim.hpp"

namespace kmrb {

class Renderer {
public:
    void init(GLFWwindow* window, vk::Instance instance,
              vk::Device device, vk::PhysicalDevice physicalDevice,
              vk::Format swapchainFormat, vk::Extent2D extent,
              const std::vector<vk::ImageView>& swapchainImageViews,
              uint32_t graphicsQueueFamily, vk::Queue graphicsQueue,
              uint32_t particleCount = 10000);
    void cleanup(vk::Device device);

    void onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                             const std::vector<vk::ImageView>& newImageViews);

    bool drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                   vk::Queue graphicsQueue, vk::Queue presentQueue,
                   GLFWwindow* window);

    void uploadParticles(vk::Device device, const std::vector<Particle>& particles);
    void setRegistry(entt::registry* reg) { registry = reg; }

    vk::RenderPass getRenderPass() const { return offscreenPass; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout; }
    const BufferManager& getBufferManager() const { return bufferManager; }
    BufferManager& getBufferManager() { return bufferManager; }
    UI& getUI() { return ui; }
    void* getShaderInstancesPtr() { return &shaderInstances; }
    void* getMeshInstancesPtr() { return &meshInstances; }
    MeshCache& getMeshCache() { return meshCache; }

    // Environment map — called by Core via UI callback
    void loadEnvMap(vk::Device device, const std::string& path) { loadEnvironmentMap(device, path); }
    void clearEnvMap(vk::Device device) { destroyEnvironmentMap(device); }

    // Destroy all mesh GPU buffers and force a full reload on next sync
    void clearMeshCache(vk::Device device);

    // Re-run init shaders on next frame (called after simulation reset)
    void requestInitDispatch() {
        for (auto& [key, inst] : shaderInstances) {
            if (inst.initPipeline) inst.initPending = true;
        }
    }

    // A single tweakable parameter extracted from shader push constants via SPIRV-Reflect.
    // The Inspector uses this to auto-generate UI widgets (sliders, checkboxes, etc.)
    // so users can tweak shader behavior without editing GLSL.
    struct ReflectedParam {
        std::string name;       // GLSL variable name, e.g. "gravity", "damping"
        uint32_t offset;        // Byte offset within push constant block
        uint32_t size;          // Size in bytes (4 for float, 12 for vec3, etc.)
        enum Type { Float, Vec2, Vec3, Vec4, Int, Bool, Mat4, Unknown } type;
    };

    // Per-entity shader instances — replaces singleton pipelines.
    // Each Pipeline entity gets its own compiled Vulkan pipelines, reflected params,
    // and live push constant data buffer that the Inspector can write into.
    struct ShaderInstance {
        entt::entity owner = entt::null;
        vk::Pipeline initPipeline = nullptr;     // One-shot: runs once to set up initial particle positions
        vk::Pipeline computePipeline = nullptr;  // Per-frame: runs every frame for simulation
        vk::Pipeline graphicsPipeline = nullptr;
        std::string initPath, computePath, vertexPath, fragmentPath;
        std::filesystem::file_time_type initModTime{}, compModTime{}, vertModTime{}, fragModTime{};
        bool initPending = false;                // True = init shader needs to dispatch (once)

        // Workgroup sizes read from the compiled SPIR-V (local_size_x in the user's GLSL).
        // The dispatch count is derived from these so users can pick any workgroup size.
        uint32_t initLocalSizeX = 256;
        uint32_t computeLocalSizeX = 256;

        // Reflection data — populated by reflectPushConstants() after SPIR-V compilation
        std::vector<ReflectedParam> reflectedParams;   // User-tweakable params (excludes engine built-ins)
        std::vector<uint8_t> pushConstantData;          // Live values written to GPU each frame
        uint32_t pushConstantSize = 0;                  // Total push constant block size in bytes
    };

    // Per-mesh-entity shader instance — simpler than particle ShaderInstance (no compute/init).
    // Each Mesh entity gets its own graphics pipeline built from user-assignable vert/frag shaders.
    struct MeshShaderInstance {
        entt::entity owner = entt::null;
        vk::Pipeline graphicsPipeline = nullptr;
        std::string vertexPath, fragmentPath;
        std::filesystem::file_time_type vertModTime{}, fragModTime{};
        bool wireframe = false;

        std::vector<ReflectedParam> reflectedParams;
        std::vector<uint8_t> pushConstantData;
        uint32_t pushConstantSize = 0;
    };

private:
    vk::PhysicalDevice physicalDevice;
    GLFWwindow* cachedWindow = nullptr;
    entt::registry* registry = nullptr;

    // ── Two Render Passes ──
    // Offscreen: particles → color+depth image (sampled by ImGui as texture)
    // Swapchain: ImGui only → presented to screen
    vk::RenderPass offscreenPass;
    vk::RenderPass swapchainPass;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline gridPipeline;        // Lines — grid rendering (always present)
    vk::Pipeline skyboxPipeline = nullptr;  // Fullscreen skybox rendering

    std::unordered_map<uint32_t, ShaderInstance> shaderInstances;
    std::unordered_map<uint32_t, MeshShaderInstance> meshInstances;
    MeshCache meshCache;

    // ── Environment Map (scene-level skybox + shader-accessible cubemap) ──
    vk::Image envCubemap = nullptr;
    vk::DeviceMemory envCubemapMemory = nullptr;
    vk::ImageView envCubemapView = nullptr;
    vk::Sampler envSampler = nullptr;
    vk::DescriptorSet envDescriptorSet = nullptr;   // Set 1: environment cubemap + IBL maps
    bool envMapLoaded = false;

    // ── IBL (image-based lighting) — precomputed maps the PBR shaders sample ──
    // Fixed-size images created once at init; contents re-baked on every HDR
    // load/clear by dispatching the engine compute shaders in shaders/engine/.
    static constexpr uint32_t IBL_IRRADIANCE_SIZE  = 32;   // diffuse map, per face
    static constexpr uint32_t IBL_PREFILTERED_SIZE = 128;  // specular map mip 0, per face
    static constexpr uint32_t IBL_PREFILTER_MIPS   = 5;    // 128 → 8, one roughness per mip
    static constexpr uint32_t IBL_BRDF_LUT_SIZE    = 512;

    vk::Image        iblIrradiance = nullptr;               // Set 1, binding 1
    vk::DeviceMemory iblIrradianceMemory = nullptr;
    vk::ImageView    iblIrradianceView = nullptr;            // cube view (sampling)
    vk::ImageView    iblIrradianceStorageView = nullptr;     // 2D-array view (compute write)
    vk::Sampler      iblIrradianceSampler = nullptr;

    vk::Image        iblPrefiltered = nullptr;               // Set 1, binding 2
    vk::DeviceMemory iblPrefilteredMemory = nullptr;
    vk::ImageView    iblPrefilteredView = nullptr;            // cube view, all mips (sampling)
    std::array<vk::ImageView, IBL_PREFILTER_MIPS> iblPrefilteredMipViews{}; // per-mip write targets
    vk::Sampler      iblPrefilteredSampler = nullptr;         // maxLod unclamped → textureLod works

    vk::Image        iblBrdfLut = nullptr;                   // Set 1, binding 3
    vk::DeviceMemory iblBrdfLutMemory = nullptr;
    vk::ImageView    iblBrdfLutView = nullptr;
    vk::Sampler      iblBrdfLutSampler = nullptr;

    // Bake pipelines — separate layout from the main one (needs storage images)
    vk::DescriptorSetLayout iblBakeSetLayout = nullptr;  // 0: samplerCube src, 1: storage image dst
    vk::PipelineLayout      iblBakePipelineLayout = nullptr;
    vk::Pipeline            iblBrdfPipeline = nullptr;
    vk::Pipeline            iblIrradiancePipeline = nullptr;
    vk::Pipeline            iblPrefilterPipeline = nullptr;

    // ── Offscreen Framebuffer (simulation viewport) ──
    vk::Image offscreenColor;
    vk::DeviceMemory offscreenColorMemory;
    vk::ImageView offscreenColorView;
    vk::Image offscreenDepth;
    vk::DeviceMemory offscreenDepthMemory;
    vk::ImageView offscreenDepthView;
    vk::Framebuffer offscreenFramebuffer;
    vk::Sampler offscreenSampler;
    vk::DescriptorSet offscreenImGuiDescriptor = nullptr; // ImGui texture handle
    vk::Format colorFormat;
    vk::Format depthFormat = vk::Format::eD32Sfloat;

    // ── Swapchain Framebuffers (ImGui only) ──
    std::vector<vk::Framebuffer> swapchainFramebuffers;

    // ── Descriptor System ──
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_COUNT> descriptorSetLayouts;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> globalDescriptorSets;

    // ── UI & Camera ──
    UI ui;
    Camera camera;

    // ── Buffer Manager ──
    BufferManager bufferManager;

    std::array<vk::DescriptorSet, 2> particleDescriptorSets;
    uint32_t particleCount = 0;
    uint32_t pingPong = 0;

    // ── Command Recording ──
    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    // ── Sync ──
    static constexpr int MAX_FRAMES_IN_FLIGHT = 2;
    std::vector<vk::Semaphore> acquireSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;

    uint32_t currentFrame = 0;
    uint32_t imageCount = 0;
    uint32_t graphicsQueueFamily = 0;
    vk::Extent2D extent;          // Swapchain size (full window)
    vk::Extent2D renderExtent;    // Offscreen framebuffer size (set in Preferences)
    float elapsedTime = 0.0f;
    float computeTime = 0.0f;

    // ── Frame timing & periodic work (members, not function statics) ──
    std::chrono::steady_clock::time_point appStartTime;   // For the shader 'time' uniform
    std::chrono::steady_clock::time_point lastFrameTime;  // For real per-frame delta
    uint32_t hotReloadCounter = 0;   // Frames since the last shader file poll
    bool firstSyncDone = false;      // Initial shader/mesh instance sync happened


    // ── Init Helpers ──
    uint32_t gridVertexCount = 0;

    // ── Light Gizmos ──
    struct GizmoDrawCmd {
        uint32_t firstVertex;
        uint32_t vertexCount;
        glm::vec4 color;
        bool selected;
    };
    std::vector<GizmoDrawCmd> gizmoDraws;
    std::vector<glm::vec3> gizmoVerts;   // Scratch buffer, reused every frame

    void updateGizmoBuffer(vk::Device device);
    void drawLightGizmos(vk::CommandBuffer cmd, uint32_t imageIndex);
    static void appendCross(std::vector<glm::vec3>& out, const glm::vec3& c, float s);
    static void appendSphereRings(std::vector<glm::vec3>& out, const glm::vec3& c, float r, int seg);
    static void appendSpotCone(std::vector<glm::vec3>& out, const glm::vec3& apex,
                               const glm::vec3& dir, float angleDeg, float range, int seg);
    static void appendDirArrows(std::vector<glm::vec3>& out, const glm::vec3& origin,
                                const glm::vec3& dir, float len);

    void createOffscreenPass(vk::Device device);
    void createGridPipeline(vk::Device device);
    void updateGridBuffer(vk::Device device);
    void createSwapchainRenderPass(vk::Device device);
    void createDescriptorSetLayouts(vk::Device device);
    void createPipelineLayout(vk::Device device);
    void createOffscreenResources(vk::Device device);
    void createSwapchainFramebuffers(vk::Device device, const std::vector<vk::ImageView>& imageViews);
    void createGlobalUBOBuffers(vk::Device device);
    void createDescriptorPool(vk::Device device);
    void createDescriptorSets(vk::Device device);
    void createParticleBuffer(vk::Device device);
    void createCommandPool(vk::Device device, uint32_t queueFamily);
    void createCommandBuffers(vk::Device device);
    void createSyncObjects(vk::Device device);

    void cleanupSwapchainResources(vk::Device device);
    void cleanupOffscreenResources(vk::Device device);
    void recordCommandBuffer(vk::CommandBuffer cmd, uint32_t imageIndex);
    void updateGlobalUBO(uint32_t imageIndex);

    // Shader instance management — syncs ECS ShaderProgramComponents to Vulkan pipelines.
    // Builders take pre-compiled SPIR-V so each shader is compiled exactly once
    // (the same SPIR-V is also fed to reflection).
    void syncShaderInstances(vk::Device device);
    void destroyShaderInstance(vk::Device device, ShaderInstance& inst);
    vk::Pipeline buildComputePipeline(vk::Device device, const std::vector<uint32_t>& spirv);
    vk::Pipeline buildGraphicsPipeline(vk::Device device, const std::vector<uint32_t>& vertSpirv,
                                       const std::vector<uint32_t>& fragSpirv);

    // Mesh instance management — syncs MeshRendererComponents to Vulkan pipelines
    void syncMeshInstances(vk::Device device);
    void destroyMeshInstance(vk::Device device, MeshShaderInstance& inst);
    vk::Pipeline buildMeshGraphicsPipeline(vk::Device device, const std::vector<uint32_t>& vertSpirv,
                                           const std::vector<uint32_t>& fragSpirv, bool wireframe);

    // SPIRV-Reflect: extract push constant members from compiled SPIR-V bytecode.
    // Populates inst.reflectedParams with user-tweakable params (skips engine built-ins like model/color).
    void reflectPushConstants(const std::vector<uint32_t>& spirv, ShaderInstance& inst);

    // Environment map — load HDR, create cubemap, bind to Set 1
    void loadEnvironmentMap(vk::Device device, const std::string& hdrPath);
    void destroyEnvironmentMap(vk::Device device);
    void createSkyboxPipeline(vk::Device device);
    void createPlaceholderCubemap(vk::Device device);  // 1x1 black cubemap for unbound state

    // IBL — images/pipelines at init (+ one-time BRDF LUT bake), then
    // bakeIBLMaps() re-convolves irradiance/prefiltered from the current envMap
    void createIBLResources(vk::Device device);
    void bakeIBLMaps(vk::Device device);
    void destroyIBLResources(vk::Device device);
    void writeIBLDescriptors(vk::Device device);  // bindings 1-3 of envDescriptorSet

    vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);
    static std::vector<char> readFile(const std::string& filename);
    static std::vector<uint32_t> compileGLSL(const std::string& sourcePath);
};

} // namespace kmrb
