#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>
#include <filesystem>

#include "kmrb_types.hpp"
#include "kmrb_buffers.hpp"

namespace kmrb {

class Renderer {
public:
    void init(vk::Device device, vk::PhysicalDevice physicalDevice,
              vk::Format swapchainFormat, vk::Extent2D extent,
              const std::vector<vk::ImageView>& swapchainImageViews,
              uint32_t graphicsQueueFamily,
              uint32_t particleCount = 10000);
    void cleanup(vk::Device device);

    // Swapchain recreation — destroys/recreates only size-dependent resources
    void onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                             const std::vector<vk::ImageView>& newImageViews);

    // Returns true if swapchain needs recreation (window resized, OUT_OF_DATE)
    bool drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                   vk::Queue graphicsQueue, vk::Queue presentQueue);

    // Upload particle data from CPU (ECS sync) to both SSBO buffers
    void uploadParticles(vk::Device device, const std::vector<Particle>& particles);

    // Set the compute shader source file to watch for hot-reload
    void setComputeShader(const std::string& path);

    vk::RenderPass getRenderPass() const { return renderPass; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout; }
    const BufferManager& getBufferManager() const { return bufferManager; }

private:
    vk::PhysicalDevice physicalDevice;

    // ── Render Pass & Pipelines ──
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;     // Shared between graphics and compute
    vk::Pipeline graphicsPipeline;
    vk::Pipeline computePipeline;          // Particle simulation
    std::vector<vk::Framebuffer> framebuffers;

    // ── Depth Buffer ──
    // Single depth image shared across frames (only one frame renders at a time per framebuffer)
    vk::Image depthImage;
    vk::DeviceMemory depthImageMemory;
    vk::ImageView depthImageView;
    vk::Format depthFormat = vk::Format::eD32Sfloat;

    // ── Descriptor System ──
    // 3 layouts following the KMRB convention (global / material / object)
    std::array<vk::DescriptorSetLayout, DESCRIPTOR_SET_COUNT> descriptorSetLayouts;
    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> globalDescriptorSets; // One per swapchain image

    // ── Buffer Manager ──
    BufferManager bufferManager;
    // Buffer names used as keys:
    //   "global_ubo_0", "global_ubo_1", ...   — per swapchain image
    //   "particle_a", "particle_b"            — double-buffered SSBOs

    std::array<vk::DescriptorSet, 2> particleDescriptorSets; // [0]: A→B, [1]: B→A
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

    // ── Shader Hot-Reload ──
    std::string computeShaderPath;                          // Source .comp file to watch
    std::filesystem::file_time_type lastShaderModTime;      // Last known modification time
    float shaderPollTimer = 0.0f;                           // Accumulator for poll interval
    static constexpr float SHADER_POLL_INTERVAL = 0.5f;    // Check every 0.5 seconds

    // ── Init Helpers ──
    void createRenderPass(vk::Device device, const RenderPassConfig& config);
    void createDescriptorSetLayouts(vk::Device device);
    void createGraphicsPipeline(vk::Device device);
    void createComputePipeline(vk::Device device);
    void createDepthResources(vk::Device device);
    void createFramebuffers(vk::Device device, const std::vector<vk::ImageView>& swapchainImageViews);
    void createGlobalUBOBuffers(vk::Device device);
    void createDescriptorPool(vk::Device device);
    void createDescriptorSets(vk::Device device);
    void createParticleBuffer(vk::Device device);
    void createCommandPool(vk::Device device, uint32_t graphicsQueueFamily);
    void createCommandBuffers(vk::Device device);
    void createSyncObjects(vk::Device device);

    void cleanupSwapchainResources(vk::Device device);
    void recordCommandBuffer(vk::CommandBuffer cmd, uint32_t imageIndex);
    void updateGlobalUBO(uint32_t imageIndex);
    void checkShaderReload(vk::Device device);

    // ── Vulkan Helpers ──
    vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);
    static std::vector<char> readFile(const std::string& filename);

    // Runtime GLSL → SPIR-V compilation via glslc subprocess
    // Returns empty vector on failure (errors printed to console)
    static std::vector<uint32_t> compileGLSL(const std::string& sourcePath);
};

} // namespace kmrb
