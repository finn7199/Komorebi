#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <filesystem>

#include "kmrb_types.hpp"
#include "kmrb_buffers.hpp"
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

    // Per-entity shader instances — replaces singleton pipelines
    struct ShaderInstance {
        entt::entity owner = entt::null;
        vk::Pipeline computePipeline = nullptr;
        vk::Pipeline graphicsPipeline = nullptr;
        std::string computePath, vertexPath, fragmentPath;
        std::filesystem::file_time_type compModTime{}, vertModTime{}, fragModTime{};
    };
    std::unordered_map<uint32_t, ShaderInstance> shaderInstances;

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
    vk::Extent2D extent;
    float elapsedTime = 0.0f;
    float computeTime = 0.0f;

    bool currentF64 = false;

    // ── Init Helpers ──
    uint32_t gridVertexCount = 0;

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

    // Shader instance management — syncs ECS ShaderProgramComponents to Vulkan pipelines
    void syncShaderInstances(vk::Device device);
    void destroyShaderInstance(vk::Device device, ShaderInstance& inst);
    vk::Pipeline buildComputePipeline(vk::Device device, const std::string& path);
    vk::Pipeline buildGraphicsPipeline(vk::Device device, const std::string& vertPath, const std::string& fragPath);

    vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);
    static std::vector<char> readFile(const std::string& filename);
    static std::vector<uint32_t> compileGLSL(const std::string& sourcePath, bool useF64 = false);
};

} // namespace kmrb
