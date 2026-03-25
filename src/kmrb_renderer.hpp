#pragma once

#include <vulkan/vulkan.hpp>
#include <vector>
#include <string>

#include "kmrb_types.hpp"

namespace kmrb {

class Renderer {
public:
    void init(vk::Device device, vk::PhysicalDevice physicalDevice,
              vk::Format swapchainFormat, vk::Extent2D extent,
              const std::vector<vk::ImageView>& swapchainImageViews,
              uint32_t graphicsQueueFamily);
    void cleanup(vk::Device device);

    // Swapchain recreation — destroys/recreates only size-dependent resources
    void onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                             const std::vector<vk::ImageView>& newImageViews);

    // Returns true if swapchain needs recreation (window resized, OUT_OF_DATE)
    bool drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                   vk::Queue graphicsQueue, vk::Queue presentQueue);

    vk::RenderPass getRenderPass() const { return renderPass; }
    vk::PipelineLayout getPipelineLayout() const { return pipelineLayout; }

private:
    vk::PhysicalDevice physicalDevice;

    // ── Render Pass & Pipeline ──
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
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

    // ── Global UBO (Set 0, Binding 0) ──
    std::vector<vk::Buffer> globalUBOBuffers;             // One per swapchain image
    std::vector<vk::DeviceMemory> globalUBOMemory;
    std::vector<void*> globalUBOMapped;                   // Persistently mapped pointers

    // ── Particle SSBO (Set 2, Binding 0) ──
    vk::Buffer particleBuffer;
    vk::DeviceMemory particleBufferMemory;
    vk::DescriptorSet particleDescriptorSet;              // Single set — same buffer every frame
    uint32_t particleCount = 0;

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

    // ── Init Helpers ──
    void createRenderPass(vk::Device device, const RenderPassConfig& config);
    void createDescriptorSetLayouts(vk::Device device);
    void createGraphicsPipeline(vk::Device device);
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

    // ── Vulkan Helpers ──
    void createBuffer(vk::Device device, vk::DeviceSize size, vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags properties, vk::Buffer& buffer, vk::DeviceMemory& memory);
    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
    vk::ShaderModule createShaderModule(vk::Device device, const std::vector<char>& code);
    static std::vector<char> readFile(const std::string& filename);
};

} // namespace kmrb
