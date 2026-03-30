#include "kmrb_ui.hpp"
#include "kmrb_renderer.hpp"
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <cmath>

namespace kmrb {

void Renderer::init(GLFWwindow* window, vk::Instance instance,
                    vk::Device device, vk::PhysicalDevice gpu,
                    vk::Format swapchainFormat, vk::Extent2D swapExtent,
                    const std::vector<vk::ImageView>& swapchainImageViews,
                    uint32_t graphicsQueueFamily, vk::Queue graphicsQueue,
                    uint32_t initialParticleCount) {
    physicalDevice = gpu;
    cachedWindow = window;
    extent = swapExtent;
    colorFormat = swapchainFormat;
    imageCount = static_cast<uint32_t>(swapchainImageViews.size());
    this->graphicsQueueFamily = graphicsQueueFamily;
    particleCount = initialParticleCount;

    bufferManager.init(device, gpu);
    camera.init(glm::vec3(0.0f, 2.0f, 5.0f), -15.0f, -90.0f);

    createOffscreenPass(device);
    createSwapchainRenderPass(device);
    createDescriptorSetLayouts(device);
    createPipelineLayout(device);
    createGridPipeline(device);
    createOffscreenResources(device);
    createSwapchainFramebuffers(device, swapchainImageViews);
    createGlobalUBOBuffers(device);
    createDescriptorPool(device);
    createDescriptorSets(device);
    createParticleBuffer(device);
    createCommandPool(device, graphicsQueueFamily);
    createCommandBuffers(device);
    createSyncObjects(device);

    ui.init(window, instance, gpu, device, graphicsQueueFamily,
            graphicsQueue, swapchainPass, imageCount);

    // Check GPU float64 support and tell the UI
    vk::PhysicalDeviceFeatures features = gpu.getFeatures();
    ui.setGPUSupportsF64(features.shaderFloat64);

    // Create ImGui texture for the offscreen color image
    offscreenImGuiDescriptor = ImGui_ImplVulkan_AddTexture(
        offscreenSampler, offscreenColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RENDER PASSES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Offscreen pass — renders particles to a texture (finalLayout = eShaderReadOnlyOptimal)
void Renderer::createOffscreenPass(vk::Device device) {
    std::array<vk::AttachmentDescription, 2> attachments = {{
        // Color
        { {}, colorFormat, vk::SampleCountFlagBits::e1,
          vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
          vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
          vk::ImageLayout::eUndefined, vk::ImageLayout::eShaderReadOnlyOptimal },
        // Depth
        { {}, depthFormat, vk::SampleCountFlagBits::e1,
          vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
          vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
          vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal }
    }};

    vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);
    vk::AttachmentReference depthRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics,
        0, nullptr, 1, &colorRef, nullptr, &depthRef);

    vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        {},
        vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    );

    vk::RenderPassCreateInfo info({},
        static_cast<uint32_t>(attachments.size()), attachments.data(),
        1, &subpass, 1, &dependency);

    offscreenPass = device.createRenderPass(info);
    kmrb::Log::info("Offscreen render pass created");
}

// Swapchain pass — ImGui only (finalLayout = ePresentSrcKHR)
void Renderer::createSwapchainRenderPass(vk::Device device) {
    vk::AttachmentDescription colorAttachment(
        {}, colorFormat, vk::SampleCountFlagBits::e1,
        vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR
    );

    vk::AttachmentReference colorRef(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics,
        0, nullptr, 1, &colorRef);

    vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        vk::PipelineStageFlagBits::eColorAttachmentOutput,
        {},
        vk::AccessFlagBits::eColorAttachmentWrite
    );

    vk::RenderPassCreateInfo info({}, 1, &colorAttachment, 1, &subpass, 1, &dependency);
    swapchainPass = device.createRenderPass(info);
    kmrb::Log::info("Swapchain render pass created");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESCRIPTOR SET LAYOUTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createDescriptorSetLayouts(vk::Device device) {
    vk::DescriptorSetLayoutBinding globalUBOBinding(
        0, vk::DescriptorType::eUniformBuffer, 1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute
    );
    vk::DescriptorSetLayoutCreateInfo globalLayoutInfo({}, 1, &globalUBOBinding);
    descriptorSetLayouts[DESCRIPTOR_SET_GLOBAL] = device.createDescriptorSetLayout(globalLayoutInfo);

    vk::DescriptorSetLayoutCreateInfo materialLayoutInfo({}, 0, nullptr);
    descriptorSetLayouts[DESCRIPTOR_SET_MATERIAL] = device.createDescriptorSetLayout(materialLayoutInfo);

    std::array<vk::DescriptorSetLayoutBinding, 2> ssboBindings = {{
        { 0, vk::DescriptorType::eStorageBuffer, 1,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute },
        { 1, vk::DescriptorType::eStorageBuffer, 1,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute }
    }};
    vk::DescriptorSetLayoutCreateInfo objectLayoutInfo(
        {}, static_cast<uint32_t>(ssboBindings.size()), ssboBindings.data());
    descriptorSetLayouts[DESCRIPTOR_SET_OBJECT] = device.createDescriptorSetLayout(objectLayoutInfo);

    kmrb::Log::info("Descriptor set layouts created (3 sets)");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PIPELINE LAYOUT (shared by all shader instances)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createPipelineLayout(vk::Device device) {
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
        0, sizeof(PushConstants));

    vk::PipelineLayoutCreateInfo layoutInfo(
        {}, static_cast<uint32_t>(descriptorSetLayouts.size()),
        descriptorSetLayouts.data(), 1, &pushRange);
    pipelineLayout = device.createPipelineLayout(layoutInfo);
    kmrb::Log::ok("Pipeline layout created");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GRID PIPELINE (line rendering)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createGridPipeline(vk::Device device) {
    auto vertCode = readFile(KMRB_SHADER_SPV_DIR "/grid.vert.spv");
    auto fragCode = readFile(KMRB_SHADER_SPV_DIR "/grid.frag.spv");

    vk::ShaderModule vertModule = createShaderModule(device, vertCode);
    vk::ShaderModule fragModule = createShaderModule(device, fragCode);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragModule, "main" }
    };

    // Vertex input — vec3 position per vertex
    vk::VertexInputBindingDescription binding(0, sizeof(glm::vec3), vk::VertexInputRate::eVertex);
    vk::VertexInputAttributeDescription attr(0, 0, vk::Format::eR32G32B32Sfloat, 0);
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 1, &binding, 1, &attr);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::eLineList, VK_FALSE);

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
        VK_FALSE, 0, 0, 0, 1.0f);

    vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);

    vk::PipelineDepthStencilStateCreateInfo depthStencil(
        {}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE);

    vk::PipelineColorBlendAttachmentState blendAttachment(VK_FALSE);
    blendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendStateCreateInfo colorBlending({}, VK_FALSE, vk::LogicOp::eCopy, 1, &blendAttachment);

    std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dynamicState(
        {}, static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data());

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, 2, shaderStages, &vertexInputInfo, &inputAssembly, nullptr,
        &viewportState, &rasterizer, &multisampling, &depthStencil,
        &colorBlending, &dynamicState, pipelineLayout, offscreenPass, 0);

    gridPipeline = device.createGraphicsPipeline(nullptr, pipelineInfo).value;
    device.destroyShaderModule(vertModule);
    device.destroyShaderModule(fragModule);
    Log::ok("Grid pipeline created");
}

// Generate grid line vertices from the first GridTag entity's properties
void Renderer::updateGridBuffer(vk::Device device) {
    if (!registry) return;

    auto gridView = registry->view<GridComponent, Transform>();
    if (gridView.size_hint() == 0) {
        gridVertexCount = 0;
        return;
    }

    auto entity = *gridView.begin();
    auto& props = gridView.get<GridComponent>(entity);

    float half = props.size / 2.0f;
    float step = props.size / static_cast<float>(props.cellCount);
    int n = props.cellCount;

    std::vector<glm::vec3> vertices;
    vertices.reserve((n + 1) * 4);

    for (int i = 0; i <= n; i++) {
        float z = -half + i * step;
        vertices.push_back(glm::vec3(-half, 0.0f, z));
        vertices.push_back(glm::vec3(half, 0.0f, z));
    }
    for (int i = 0; i <= n; i++) {
        float x = -half + i * step;
        vertices.push_back(glm::vec3(x, 0.0f, -half));
        vertices.push_back(glm::vec3(x, 0.0f, half));
    }

    gridVertexCount = static_cast<uint32_t>(vertices.size());
    vk::DeviceSize bufSize = sizeof(glm::vec3) * gridVertexCount;

    // Allocate max-size buffer once, re-upload data each update
    // Max: 100 cells × 2 axes × 2 verts per line = 808 verts
    constexpr vk::DeviceSize maxGridBufSize = sizeof(glm::vec3) * 808;

    if (!bufferManager.exists("grid_lines")) {
        bufferManager.createBuffer("grid_lines", maxGridBufSize,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }

    bufferManager.upload("grid_lines", vertices.data(), bufSize);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// OFFSCREEN RESOURCES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createOffscreenResources(vk::Device device) {
    auto findMem = [&](vk::MemoryRequirements reqs, vk::MemoryPropertyFlags props) -> uint32_t {
        auto memProps = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((reqs.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("KMRB: No suitable memory type");
    };

    // Color image — sampled by ImGui as a texture
    vk::ImageCreateInfo colorInfo({}, vk::ImageType::e2D, colorFormat,
        vk::Extent3D(extent.width, extent.height, 1), 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled,
        vk::SharingMode::eExclusive);

    offscreenColor = device.createImage(colorInfo);
    auto colorReqs = device.getImageMemoryRequirements(offscreenColor);
    offscreenColorMemory = device.allocateMemory(
        vk::MemoryAllocateInfo(colorReqs.size, findMem(colorReqs, vk::MemoryPropertyFlagBits::eDeviceLocal)));
    device.bindImageMemory(offscreenColor, offscreenColorMemory, 0);

    offscreenColorView = device.createImageView(vk::ImageViewCreateInfo(
        {}, offscreenColor, vk::ImageViewType::e2D, colorFormat, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));

    // Depth image
    vk::ImageCreateInfo depthInfo({}, vk::ImageType::e2D, depthFormat,
        vk::Extent3D(extent.width, extent.height, 1), 1, 1,
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment, vk::SharingMode::eExclusive);

    offscreenDepth = device.createImage(depthInfo);
    auto depthReqs = device.getImageMemoryRequirements(offscreenDepth);
    offscreenDepthMemory = device.allocateMemory(
        vk::MemoryAllocateInfo(depthReqs.size, findMem(depthReqs, vk::MemoryPropertyFlagBits::eDeviceLocal)));
    device.bindImageMemory(offscreenDepth, offscreenDepthMemory, 0);

    offscreenDepthView = device.createImageView(vk::ImageViewCreateInfo(
        {}, offscreenDepth, vk::ImageViewType::e2D, depthFormat, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)));

    // Sampler for ImGui to read the color attachment
    vk::SamplerCreateInfo samplerInfo({}, vk::Filter::eLinear, vk::Filter::eLinear,
        vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge);
    offscreenSampler = device.createSampler(samplerInfo);

    // Offscreen framebuffer
    vk::ImageView fbAttachments[] = { offscreenColorView, offscreenDepthView };
    offscreenFramebuffer = device.createFramebuffer(vk::FramebufferCreateInfo(
        {}, offscreenPass, 2, fbAttachments, extent.width, extent.height, 1));

    kmrb::Log::info("Offscreen framebuffer created (" + std::to_string(extent.width) + "x" + std::to_string(extent.height) + ")");
}

void Renderer::cleanupOffscreenResources(vk::Device device) {
    device.destroyFramebuffer(offscreenFramebuffer);
    device.destroySampler(offscreenSampler);
    device.destroyImageView(offscreenColorView);
    device.destroyImage(offscreenColor);
    device.freeMemory(offscreenColorMemory);
    device.destroyImageView(offscreenDepthView);
    device.destroyImage(offscreenDepth);
    device.freeMemory(offscreenDepthMemory);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SWAPCHAIN FRAMEBUFFERS (ImGui only — no depth needed)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createSwapchainFramebuffers(vk::Device device,
                                           const std::vector<vk::ImageView>& imageViews) {
    swapchainFramebuffers.resize(imageViews.size());
    for (size_t i = 0; i < imageViews.size(); i++) {
        vk::ImageView attachments[] = { imageViews[i] };
        swapchainFramebuffers[i] = device.createFramebuffer(vk::FramebufferCreateInfo(
            {}, swapchainPass, 1, attachments, extent.width, extent.height, 1));
    }
    kmrb::Log::info("Swapchain framebuffers created (" + std::to_string(swapchainFramebuffers.size()) + ")");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BUFFERS, DESCRIPTORS, COMMANDS, SYNC (same as before)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createGlobalUBOBuffers(vk::Device device) {
    for (uint32_t i = 0; i < imageCount; i++) {
        std::string name = "global_ubo_" + std::to_string(i);
        bufferManager.createBuffer(name, sizeof(GlobalUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            true);
    }
    kmrb::Log::info("Global UBO buffers created (" + std::to_string(imageCount) + ")");
}

void Renderer::createDescriptorPool(vk::Device device) {
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eUniformBuffer, imageCount },
        { vk::DescriptorType::eStorageBuffer, 8 },
        { vk::DescriptorType::eCombinedImageSampler, imageCount * 8 }
    };

    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        imageCount * 10,
        static_cast<uint32_t>(poolSizes.size()), poolSizes.data());

    descriptorPool = device.createDescriptorPool(poolInfo);
    kmrb::Log::info("Descriptor pool created");
}

void Renderer::createDescriptorSets(vk::Device device) {
    std::vector<vk::DescriptorSetLayout> layouts(imageCount, descriptorSetLayouts[DESCRIPTOR_SET_GLOBAL]);
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool,
        static_cast<uint32_t>(layouts.size()), layouts.data());
    globalDescriptorSets = device.allocateDescriptorSets(allocInfo);

    for (uint32_t i = 0; i < imageCount; i++) {
        std::string name = "global_ubo_" + std::to_string(i);
        vk::DescriptorBufferInfo bufferInfo(bufferManager.getBuffer(name), 0, sizeof(GlobalUBO));
        vk::WriteDescriptorSet write(globalDescriptorSets[i], 0, 0, 1,
            vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfo);
        device.updateDescriptorSets(write, nullptr);
    }

    std::array<vk::DescriptorSetLayout, 2> particleLayouts = {
        descriptorSetLayouts[DESCRIPTOR_SET_OBJECT], descriptorSetLayouts[DESCRIPTOR_SET_OBJECT]
    };
    vk::DescriptorSetAllocateInfo particleAllocInfo(descriptorPool,
        static_cast<uint32_t>(particleLayouts.size()), particleLayouts.data());
    auto particleSets = device.allocateDescriptorSets(particleAllocInfo);
    particleDescriptorSets[0] = particleSets[0];
    particleDescriptorSets[1] = particleSets[1];

    kmrb::Log::info("Descriptor sets allocated");
}

void Renderer::createParticleBuffer(vk::Device device) {
    if (particleCount == 0) particleCount = 1;
    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;
    const char* names[] = { "particle_a", "particle_b" };

    for (int i = 0; i < 2; i++) {
        bufferManager.createBuffer(names[i], bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        bufferManager.setElementInfo(names[i], particleCount, sizeof(Particle));
    }

    for (int i = 0; i < 2; i++) {
        vk::DescriptorBufferInfo inputInfo(bufferManager.getBuffer(names[i]), 0, bufferSize);
        vk::DescriptorBufferInfo outputInfo(bufferManager.getBuffer(names[1 - i]), 0, bufferSize);
        std::array<vk::WriteDescriptorSet, 2> writes = {{
            { particleDescriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputInfo },
            { particleDescriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputInfo }
        }};
        device.updateDescriptorSets(writes, nullptr);
    }

    kmrb::Log::info("Particle SSBOs allocated (2x " + std::to_string(particleCount) + ")");
}

void Renderer::uploadParticles(vk::Device device, const std::vector<Particle>& particles) {
    particleCount = static_cast<uint32_t>(particles.size());
    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;
    bufferManager.upload("particle_a", particles.data(), bufferSize);
    bufferManager.upload("particle_b", particles.data(), bufferSize);
    kmrb::Log::ok("Particles uploaded (" + std::to_string(particleCount) + " from ECS)");
}


void Renderer::createCommandPool(vk::Device device, uint32_t queueFamily) {
    commandPool = device.createCommandPool(vk::CommandPoolCreateInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamily));
    kmrb::Log::info("Command pool created");
}

void Renderer::createCommandBuffers(vk::Device device) {
    commandBuffers = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, imageCount));
    kmrb::Log::info("Command buffers allocated (" + std::to_string(commandBuffers.size()) + ")");
}

void Renderer::createSyncObjects(vk::Device device) {
    acquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(imageCount);

    vk::SemaphoreCreateInfo semInfo{};
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        acquireSemaphores[i] = device.createSemaphore(semInfo);
        inFlightFences[i] = device.createFence(fenceInfo);
    }
    for (uint32_t i = 0; i < imageCount; i++)
        renderFinishedSemaphores[i] = device.createSemaphore(semInfo);

    imagesInFlight.assign(imageCount, nullptr);
    kmrb::Log::info("Sync objects created");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UBO UPDATE (uses camera)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::updateGlobalUBO(uint32_t imageIndex) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(currentTime - startTime).count();

    // Read FOV/near/far from the active camera entity, fall back to defaults
    float fov = 45.0f;
    float nearPlane = 0.1f;
    float farPlane = 100.0f;

    if (registry) {
        auto camView = registry->view<CameraComponent>();
        for (auto entity : camView) {
            auto& props = camView.get<CameraComponent>(entity);
            if (props.active) {
                fov = props.fov;
                nearPlane = props.nearPlane;
                farPlane = props.farPlane;
                break;
            }
        }
    }

    GlobalUBO ubo{};
    ubo.view = camera.getViewMatrix();
    ubo.proj = glm::perspective(glm::radians(fov),
        static_cast<float>(extent.width) / static_cast<float>(extent.height), nearPlane, farPlane);
    ubo.proj[1][1] *= -1;
    ubo.cameraPos = glm::vec4(camera.position, 1.0f);
    float rawDt = time - elapsedTime;
    ubo.deltaTime = std::min(rawDt, 0.033f); // Cap at ~30fps — prevents physics explosion after pauses
    ubo.time = time;
    elapsedTime = time;

    std::string name = "global_ubo_" + std::to_string(imageIndex);
    memcpy(bufferManager.getMappedData(name), &ubo, sizeof(ubo));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMAND RECORDING
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::recordCommandBuffer(vk::CommandBuffer cmd, uint32_t imageIndex) {
    cmd.begin(vk::CommandBufferBeginInfo{});

    // Find the active pipeline's shader instance
    ShaderInstance* activeInstance = nullptr;
    if (registry) {
        auto view = registry->view<PipelineComponent, ShaderProgramComponent>();
        for (auto entity : view) {
            uint32_t key = static_cast<uint32_t>(entity);
            auto it = shaderInstances.find(key);
            if (it != shaderInstances.end()) {
                activeInstance = &it->second;
                break;
            }
        }
    }

    // ── COMPUTE ──
    if (activeInstance && activeInstance->computePipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, activeInstance->computePipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr);
        cmd.dispatch((particleCount + 255) / 256, 1, 1);

        vk::Buffer outputBuffer = bufferManager.getBuffer(pingPong == 0 ? "particle_b" : "particle_a");
        vk::BufferMemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, outputBuffer, 0, VK_WHOLE_SIZE);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexShader, {}, nullptr, barrier, nullptr);
    }

    // ── OFFSCREEN PASS ──
    std::array<vk::ClearValue, 2> offscreenClear = {
        vk::ClearValue(vk::ClearColorValue(std::array<float,4>{0.04f, 0.04f, 0.06f, 1.0f})),
        vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))
    };

    cmd.beginRenderPass(vk::RenderPassBeginInfo(
        offscreenPass, offscreenFramebuffer,
        vk::Rect2D({0,0}, extent),
        static_cast<uint32_t>(offscreenClear.size()), offscreenClear.data()),
        vk::SubpassContents::eInline);

    vk::Viewport viewport(0, 0, (float)extent.width, (float)extent.height, 0, 1);
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D({0,0}, extent));

    // Draw particles if we have an active graphics pipeline
    if (activeInstance && activeInstance->graphicsPipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, activeInstance->graphicsPipeline);

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr);

        PushConstants push{};
        push.model = glm::mat4(1.0f);
        push.color = glm::vec4(1.0f);
        cmd.pushConstants(pipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &push);

        cmd.draw(particleCount, 1, 0, 0);
    }

    // ── GRID LINES ──
    if (gridVertexCount > 0 && bufferManager.exists("grid_lines") && registry) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, gridPipeline);
        cmd.setViewport(0, viewport);
        cmd.setScissor(0, vk::Rect2D({0,0}, extent));

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);

        // Get grid entity position and color for push constants
        auto gridView = registry->view<GridComponent, Transform>();
        for (auto entity : gridView) {
            auto& gridTransform = gridView.get<Transform>(entity);
            auto& gridProps = gridView.get<GridComponent>(entity);

            PushConstants gridPush{};
            gridPush.model = glm::translate(glm::mat4(1.0f), gridTransform.position);
            gridPush.color = gridProps.color;
            cmd.pushConstants(pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                0, sizeof(PushConstants), &gridPush);

            vk::Buffer gridBuf = bufferManager.getBuffer("grid_lines");
            vk::DeviceSize offset = 0;
            cmd.bindVertexBuffers(0, gridBuf, offset);
            cmd.draw(gridVertexCount, 1, 0, 0);
            break; // Only render first grid for now
        }
    }

    cmd.endRenderPass();

    // ── SWAPCHAIN PASS: ImGui only ──
    vk::ClearValue swapClear(vk::ClearColorValue(std::array<float,4>{0.055f, 0.051f, 0.043f, 1.0f}));
    cmd.beginRenderPass(vk::RenderPassBeginInfo(
        swapchainPass, swapchainFramebuffers[imageIndex],
        vk::Rect2D({0,0}, extent), 1, &swapClear),
        vk::SubpassContents::eInline);

    ui.render(cmd);

    cmd.endRenderPass();
    cmd.end();

    pingPong = 1 - pingPong;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DRAW FRAME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool Renderer::drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                         vk::Queue graphicsQueue, vk::Queue presentQueue,
                         GLFWwindow* window) {
    static uint32_t frameCounter = 0;
    if (++frameCounter >= 30) {
        frameCounter = 0;
        syncShaderInstances(device);
        updateGridBuffer(device);
    }
    static bool initialized = false;
    if (!initialized && registry) {
        syncShaderInstances(device);
        updateGridBuffer(device);
        initialized = true;
    }

    // Camera — two-way sync with active camera entity
    camera.viewportHovered = ui.isViewportHovered();
    camera.update(window, 0.016f); // Fixed dt for input (actual physics dt is in UBO)

    if (registry) {
        auto camView = registry->view<CameraComponent, Transform>();
        for (auto entity : camView) {
            auto& cam = camView.get<CameraComponent>(entity);
            if (!cam.active) continue;

            auto& t = camView.get<Transform>(entity);

            if (camera.isUserControlling) {
                t.position = camera.position;
                t.rotation = glm::vec3(camera.pitch, camera.yaw, camera.roll);
            } else {
                camera.position = t.position;
                camera.pitch = t.rotation.x;
                camera.yaw = t.rotation.y;
                camera.roll = t.rotation.z;
            }
            break;
        }
    }

    // F64 toggle — mark all shader instances dirty to recompile
    bool wantF64 = ui.isF64Enabled();
    if (wantF64 != currentF64) {
        currentF64 = wantF64;
        if (registry) {
            auto view = registry->view<ShaderProgramComponent>();
            for (auto e : view) view.get<ShaderProgramComponent>(e).dirty = true;
        }
    }

    (void)device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    try {
        auto [result, idx] = device.acquireNextImageKHR(
            swapchain, UINT64_MAX, acquireSemaphores[currentFrame], nullptr);
        imageIndex = idx;
    } catch (vk::OutOfDateKHRError&) {
        return true;
    }

    if (imagesInFlight[imageIndex]) {
        (void)device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];
    device.resetFences(inFlightFences[currentFrame]);

    updateGlobalUBO(imageIndex);

    // ImGui frame
    ui.beginFrame();
    ui.drawEditorLayout(offscreenImGuiDescriptor, extent, particleCount,
                        ImGui::GetIO().Framerate, computeTime,
                        bufferManager.getAllBuffers());
    ui.endFrame();

    commandBuffers[imageIndex].reset();
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex);

    vk::Semaphore waitSems[] = { acquireSemaphores[currentFrame] };
    vk::Semaphore signalSems[] = { renderFinishedSemaphores[imageIndex] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo(1, waitSems, waitStages,
        1, &commandBuffers[imageIndex], 1, signalSems);
    graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    vk::PresentInfoKHR presentInfo(1, signalSems, 1, &swapchain, &imageIndex);
    bool needsRecreate = false;
    try {
        if (presentQueue.presentKHR(presentInfo) == vk::Result::eSuboptimalKHR)
            needsRecreate = true;
    } catch (vk::OutOfDateKHRError&) {
        needsRecreate = true;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    return needsRecreate;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SWAPCHAIN RECREATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::cleanupSwapchainResources(vk::Device device) {
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        device.destroySemaphore(acquireSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
    for (uint32_t i = 0; i < imageCount; i++)
        device.destroySemaphore(renderFinishedSemaphores[i]);

    device.destroyCommandPool(commandPool);
    for (auto& fb : swapchainFramebuffers)
        device.destroyFramebuffer(fb);

    cleanupOffscreenResources(device);

    for (uint32_t i = 0; i < imageCount; i++)
        bufferManager.destroyBuffer("global_ubo_" + std::to_string(i));

    device.destroyDescriptorPool(descriptorPool);
}

void Renderer::onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                                   const std::vector<vk::ImageView>& newImageViews) {
    cleanupSwapchainResources(device);

    extent = newExtent;
    imageCount = static_cast<uint32_t>(newImageViews.size());
    currentFrame = 0;
    pingPong = 0;

    createOffscreenResources(device);
    createSwapchainFramebuffers(device, newImageViews);
    createGlobalUBOBuffers(device);
    createDescriptorPool(device);
    createDescriptorSets(device);

    // Re-wire particle buffers to new descriptor sets
    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;
    const char* names[] = { "particle_a", "particle_b" };
    for (int i = 0; i < 2; i++) {
        vk::DescriptorBufferInfo inputInfo(bufferManager.getBuffer(names[i]), 0, bufferSize);
        vk::DescriptorBufferInfo outputInfo(bufferManager.getBuffer(names[1 - i]), 0, bufferSize);
        std::array<vk::WriteDescriptorSet, 2> writes = {{
            { particleDescriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputInfo },
            { particleDescriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputInfo }
        }};
        device.updateDescriptorSets(writes, nullptr);
    }

    createCommandPool(device, graphicsQueueFamily);
    createCommandBuffers(device);
    createSyncObjects(device);

    ui.onSwapchainRecreate(imageCount);

    // Recreate ImGui texture for new offscreen image
    offscreenImGuiDescriptor = ImGui_ImplVulkan_AddTexture(
        offscreenSampler, offscreenColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    kmrb::Log::info("Renderer recreated for ");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SHADER INSTANCE MANAGEMENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vk::Pipeline Renderer::buildComputePipeline(vk::Device device, const std::string& path) {
    auto spirv = compileGLSL(path, currentF64);
    if (spirv.empty()) {
        Log::error("Failed to compile: " + path);
        return nullptr;
    }
    vk::ShaderModuleCreateInfo moduleInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    vk::ShaderModule compModule = device.createShaderModule(moduleInfo);
    vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, compModule, "main");
    auto result = device.createComputePipeline(nullptr,
        vk::ComputePipelineCreateInfo({}, stageInfo, pipelineLayout));
    device.destroyShaderModule(compModule);
    Log::ok("Compiled compute: " + std::filesystem::path(path).filename().string());
    return result.value;
}

vk::Pipeline Renderer::buildGraphicsPipeline(vk::Device device,
                                              const std::string& vertPath,
                                              const std::string& fragPath) {
    auto vertSpirv = compileGLSL(vertPath);
    auto fragSpirv = compileGLSL(fragPath);
    if (vertSpirv.empty() || fragSpirv.empty()) {
        Log::error("Failed to compile graphics shaders");
        return nullptr;
    }

    vk::ShaderModule vertModule = device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, vertSpirv.size() * sizeof(uint32_t), vertSpirv.data()));
    vk::ShaderModule fragModule = device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, fragSpirv.size() * sizeof(uint32_t), fragSpirv.data()));

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragModule, "main" }
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 0, nullptr, 0, nullptr);
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::ePointList, VK_FALSE);
    vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
        VK_FALSE, 0, 0, 0, 1.0f);
    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, vk::SampleCountFlagBits::e1, VK_FALSE);
    vk::PipelineDepthStencilStateCreateInfo depthStencil(
        {}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE);
    vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE);
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
    vk::PipelineColorBlendStateCreateInfo colorBlending(
        {}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment);
    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport, vk::DynamicState::eScissor };
    vk::PipelineDynamicStateCreateInfo dynamicState(
        {}, static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data());

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {}, 2, shaderStages, &vertexInputInfo, &inputAssembly, nullptr,
        &viewportState, &rasterizer, &multisampling, &depthStencil,
        &colorBlending, &dynamicState, pipelineLayout, offscreenPass, 0);

    auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
    device.destroyShaderModule(vertModule);
    device.destroyShaderModule(fragModule);
    Log::ok("Compiled graphics: " + std::filesystem::path(vertPath).filename().string()
            + " + " + std::filesystem::path(fragPath).filename().string());
    return result.value;
}

void Renderer::destroyShaderInstance(vk::Device device, ShaderInstance& inst) {
    if (inst.computePipeline) { device.destroyPipeline(inst.computePipeline); inst.computePipeline = nullptr; }
    if (inst.graphicsPipeline) { device.destroyPipeline(inst.graphicsPipeline); inst.graphicsPipeline = nullptr; }
}

void Renderer::syncShaderInstances(vk::Device device) {
    if (!registry) return;
    namespace fs = std::filesystem;

    auto view = registry->view<ShaderProgramComponent>();
    for (auto entity : view) {
        auto& shader = view.get<ShaderProgramComponent>(entity);
        uint32_t key = static_cast<uint32_t>(entity);

        auto it = shaderInstances.find(key);

        // Check for hot-reload (file modification)
        if (it != shaderInstances.end() && !shader.dirty) {
            auto& inst = it->second;
            auto checkMod = [](const std::string& path, fs::file_time_type& lastMod) -> bool {
                if (path.empty() || !fs::exists(path)) return false;
                auto mod = fs::last_write_time(path);
                if (mod != lastMod) { lastMod = mod; return true; }
                return false;
            };
            if (checkMod(inst.computePath, inst.compModTime) ||
                checkMod(inst.vertexPath, inst.vertModTime) ||
                checkMod(inst.fragmentPath, inst.fragModTime)) {
                shader.dirty = true;
                Log::info("Shader change detected, recompiling...");
            }
        }

        if (!shader.dirty) continue;

        device.waitIdle();

        if (it != shaderInstances.end()) {
            destroyShaderInstance(device, it->second);
        }

        ShaderInstance inst;
        inst.owner = entity;
        inst.computePath = shader.computePath;
        inst.vertexPath = shader.vertexPath;
        inst.fragmentPath = shader.fragmentPath;

        if (!shader.computePath.empty() && fs::exists(shader.computePath)) {
            inst.computePipeline = buildComputePipeline(device, shader.computePath);
            inst.compModTime = fs::last_write_time(shader.computePath);
        }

        if (!shader.vertexPath.empty() && !shader.fragmentPath.empty() &&
            fs::exists(shader.vertexPath) && fs::exists(shader.fragmentPath)) {
            inst.graphicsPipeline = buildGraphicsPipeline(device, shader.vertexPath, shader.fragmentPath);
            inst.vertModTime = fs::last_write_time(shader.vertexPath);
            inst.fragModTime = fs::last_write_time(shader.fragmentPath);
        }

        shaderInstances[key] = inst;
        shader.dirty = false;
    }

    // Clean up orphaned instances
    std::vector<uint32_t> toRemove;
    for (auto& [key, inst] : shaderInstances) {
        if (!registry->valid(inst.owner)) {
            device.waitIdle();
            destroyShaderInstance(device, inst);
            toRemove.push_back(key);
        }
    }
    for (auto key : toRemove) shaderInstances.erase(key);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

std::vector<uint32_t> Renderer::compileGLSL(const std::string& sourcePath, bool useF64) {
    std::string outputPath = sourcePath + ".tmp.spv";
    std::string cmd = "glslc --target-env=vulkan1.3 -O";
    if (useF64) cmd += " -DUSE_F64";
    cmd += " \"" + sourcePath + "\" -o \"" + outputPath + "\" 2>&1";

    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) { kmrb::Log::error("Failed to run glslc"); return {}; }

    std::string compilerOutput;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) compilerOutput += buf;
    int exitCode = _pclose(pipe);

    if (exitCode != 0) {
        kmrb::Log::error("Shader compilation error:\n");
        std::filesystem::remove(outputPath);
        return {};
    }

    std::ifstream file(outputPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) { kmrb::Log::error("Failed to read compiled SPIR-V"); return {}; }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(spirv.data()), fileSize);
    file.close();
    std::filesystem::remove(outputPath);
    return spirv;
}

vk::ShaderModule Renderer::createShaderModule(vk::Device device, const std::vector<char>& code) {
    return device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data())));
}

std::vector<char> Renderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("KMRB: Failed to open file: " + filename);
    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(fileSize);
    file.seekg(0);
    file.read(buffer.data(), fileSize);
    return buffer;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLEANUP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::cleanup(vk::Device device) {
    ui.cleanup(device);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        device.destroySemaphore(acquireSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
    for (uint32_t i = 0; i < imageCount; i++)
        device.destroySemaphore(renderFinishedSemaphores[i]);

    device.destroyCommandPool(commandPool);
    for (auto& fb : swapchainFramebuffers)
        device.destroyFramebuffer(fb);

    cleanupOffscreenResources(device);
    bufferManager.cleanup();
    device.destroyDescriptorPool(descriptorPool);
    for (auto& layout : descriptorSetLayouts)
        device.destroyDescriptorSetLayout(layout);

    for (auto& [key, inst] : shaderInstances) destroyShaderInstance(device, inst);
    shaderInstances.clear();
    device.destroyPipeline(gridPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(offscreenPass);
    device.destroyRenderPass(swapchainPass);
}

} // namespace kmrb
