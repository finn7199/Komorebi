#include "kmrb_renderer.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstdlib>
#include <cmath>

namespace kmrb {

void Renderer::init(vk::Device device, vk::PhysicalDevice gpu,
                    vk::Format swapchainFormat, vk::Extent2D swapExtent,
                    const std::vector<vk::ImageView>& swapchainImageViews,
                    uint32_t graphicsQueueFamily,
                    uint32_t initialParticleCount) {
    physicalDevice = gpu;
    extent = swapExtent;
    imageCount = static_cast<uint32_t>(swapchainImageViews.size());
    this->graphicsQueueFamily = graphicsQueueFamily;
    particleCount = initialParticleCount;

    bufferManager.init(device, gpu);

    RenderPassConfig config{};
    config.colorFormat = swapchainFormat;
    config.hasDepth = true;
    config.depthFormat = depthFormat;

    createRenderPass(device, config);
    createDescriptorSetLayouts(device);
    createGraphicsPipeline(device);
    createComputePipeline(device);
    createDepthResources(device);
    createFramebuffers(device, swapchainImageViews);
    createGlobalUBOBuffers(device);
    createDescriptorPool(device);
    createDescriptorSets(device);
    createParticleBuffer(device);
    createCommandPool(device, graphicsQueueFamily);
    createCommandBuffers(device);
    createSyncObjects(device);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RENDER PASS (parameterized - can create shadow/gbuffer passes with different configs)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createRenderPass(vk::Device device, const RenderPassConfig& config) {
    std::vector<vk::AttachmentDescription> attachments;
    std::vector<vk::AttachmentReference> colorRefs;
    vk::AttachmentReference depthRef;

    // Color attachment
    attachments.push_back(vk::AttachmentDescription(
        {}, config.colorFormat, config.samples,
        config.loadOp, config.storeOp,
        vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
        config.initialLayout, config.finalLayout
    ));
    colorRefs.push_back({ 0, vk::ImageLayout::eColorAttachmentOptimal });

    // Depth attachment (V2: shadow maps, depth testing)
    if (config.hasDepth) {
        attachments.push_back(vk::AttachmentDescription(
            {}, config.depthFormat, config.samples,
            vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eDontCare,
            vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare,
            vk::ImageLayout::eUndefined, vk::ImageLayout::eDepthStencilAttachmentOptimal
        ));
        depthRef = { 1, vk::ImageLayout::eDepthStencilAttachmentOptimal };
    }

    vk::SubpassDescription subpass(
        {}, vk::PipelineBindPoint::eGraphics,
        0, nullptr,
        static_cast<uint32_t>(colorRefs.size()), colorRefs.data(),
        nullptr,
        config.hasDepth ? &depthRef : nullptr
    );

    vk::SubpassDependency dependency(
        VK_SUBPASS_EXTERNAL, 0,
        vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        {},
        vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    );

    vk::RenderPassCreateInfo info(
        {},
        static_cast<uint32_t>(attachments.size()), attachments.data(),
        1, &subpass,
        1, &dependency
    );

    renderPass = device.createRenderPass(info);
    std::cout << "[KMRB] Render pass created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESCRIPTOR SET LAYOUTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Set 0 (Global):   camera UBO — used by every shader
// Set 1 (Material): empty now, V2 adds PBR textures + params
// Set 2 (Object):   empty now, V2 adds per-object transforms + instance data

void Renderer::createDescriptorSetLayouts(vk::Device device) {
    // Set 0: Global — UBO at binding 0
    vk::DescriptorSetLayoutBinding globalUBOBinding(
        0,
        vk::DescriptorType::eUniformBuffer,
        1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute
    );

    vk::DescriptorSetLayoutCreateInfo globalLayoutInfo({}, 1, &globalUBOBinding);
    descriptorSetLayouts[DESCRIPTOR_SET_GLOBAL] = device.createDescriptorSetLayout(globalLayoutInfo);

    // Set 1: Material — empty for V1
    vk::DescriptorSetLayoutCreateInfo materialLayoutInfo({}, 0, nullptr);
    descriptorSetLayouts[DESCRIPTOR_SET_MATERIAL] = device.createDescriptorSetLayout(materialLayoutInfo);

    // Set 2: Particle SSBOs — double-buffered
    //   Binding 0 = input  (compute reads, vertex reads)
    //   Binding 1 = output (compute writes)
    std::array<vk::DescriptorSetLayoutBinding, 2> ssboBindings = {{
        { 0, vk::DescriptorType::eStorageBuffer, 1,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute },  // Input: compute reads, vertex reads
        { 1, vk::DescriptorType::eStorageBuffer, 1,
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute }   // Output: compute writes, vertex reads
    }};

    vk::DescriptorSetLayoutCreateInfo objectLayoutInfo(
        {}, static_cast<uint32_t>(ssboBindings.size()), ssboBindings.data()
    );
    descriptorSetLayouts[DESCRIPTOR_SET_OBJECT] = device.createDescriptorSetLayout(objectLayoutInfo);

    std::cout << "[KMRB] Descriptor set layouts created (3 sets reserved)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GRAPHICS PIPELINE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createGraphicsPipeline(vk::Device device) {
    auto vertCode = readFile(KMRB_SHADER_SPV_DIR "/particle.vert.spv");
    auto fragCode = readFile(KMRB_SHADER_SPV_DIR "/particle.frag.spv");

    vk::ShaderModule vertModule = createShaderModule(device, vertCode);
    vk::ShaderModule fragModule = createShaderModule(device, fragCode);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragModule, "main" }
    };

    // Vertex input — currently hardcoded in shader, but format is ready for vertex buffers
    // When we add vertex buffers, swap these out for Vertex::getBindingDescription() etc.
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, 0, nullptr, 0, nullptr);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::ePointList, VK_FALSE
    );

    vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);

    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {},
        VK_FALSE, VK_FALSE,
        vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone,
        vk::FrontFace::eCounterClockwise,
        VK_FALSE, 0.0f, 0.0f, 0.0f,
        1.0f
    );

    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, vk::SampleCountFlagBits::e1, VK_FALSE
    );

    vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE);
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlending(
        {}, VK_FALSE, vk::LogicOp::eCopy, 1, &colorBlendAttachment
    );

    std::vector<vk::DynamicState> dynamicStates = {
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor
    };
    vk::PipelineDynamicStateCreateInfo dynamicState(
        {}, static_cast<uint32_t>(dynamicStates.size()), dynamicStates.data()
    );

    // Push constant range — shared across all shader stages
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
        0, sizeof(PushConstants)
    );

    // Pipeline layout with all 3 descriptor set layouts + push constants
    vk::PipelineLayoutCreateInfo layoutInfo(
        {},
        static_cast<uint32_t>(descriptorSetLayouts.size()),
        descriptorSetLayouts.data(),
        1, &pushRange
    );
    pipelineLayout = device.createPipelineLayout(layoutInfo);

    // Depth testing — compare each fragment's depth, discard if behind existing geometry
    vk::PipelineDepthStencilStateCreateInfo depthStencil(
        {},
        VK_TRUE,                               // Enable depth test
        VK_TRUE,                               // Write to depth buffer
        vk::CompareOp::eLess,                  // Closer fragments win (smaller z = closer)
        VK_FALSE,                              // Depth bounds test (disabled)
        VK_FALSE                               // Stencil test (disabled)
    );

    vk::GraphicsPipelineCreateInfo pipelineInfo(
        {},
        2, shaderStages,
        &vertexInputInfo,
        &inputAssembly,
        nullptr,
        &viewportState,
        &rasterizer,
        &multisampling,
        &depthStencil,
        &colorBlending,
        &dynamicState,
        pipelineLayout,
        renderPass, 0
    );

    auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
    graphicsPipeline = result.value;

    device.destroyShaderModule(vertModule);
    device.destroyShaderModule(fragModule);

    std::cout << "[KMRB] Graphics pipeline created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMPUTE PIPELINE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compute pipelines are much simpler than graphics — just one shader stage, no
// rasterizer/blend/vertex state. Same pipeline layout so it shares descriptor sets.

void Renderer::createComputePipeline(vk::Device device) {
    if (computeShaderPath.empty()) computeShaderPath = "shaders/gravity.comp";

    auto spirv = compileGLSL(computeShaderPath);
    if (spirv.empty()) {
        throw std::runtime_error("KMRB: Failed to compile compute shader: " + computeShaderPath);
    }

    // Track modification time for hot-reload
    if (std::filesystem::exists(computeShaderPath)) {
        lastShaderModTime = std::filesystem::last_write_time(computeShaderPath);
    }

    vk::ShaderModuleCreateInfo moduleInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    vk::ShaderModule compModule = device.createShaderModule(moduleInfo);

    vk::PipelineShaderStageCreateInfo stageInfo(
        {}, vk::ShaderStageFlagBits::eCompute, compModule, "main"
    );

    vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout);

    auto result = device.createComputePipeline(nullptr, pipelineInfo);
    computePipeline = result.value;

    device.destroyShaderModule(compModule);
    std::cout << "[KMRB] Compute pipeline created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DEPTH BUFFER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// In OpenGL: glEnable(GL_DEPTH_TEST) and the driver handled the rest.
// In Vulkan: you create the depth image, allocate its memory, create a view,
// attach it to the render pass, and enable depth testing in the pipeline.

void Renderer::createDepthResources(vk::Device device) {
    // Create the depth image — same size as swapchain, used as a depth attachment
    vk::ImageCreateInfo imageInfo(
        {},
        vk::ImageType::e2D,
        depthFormat,
        vk::Extent3D(extent.width, extent.height, 1),
        1, 1,                                      // Mip levels, array layers
        vk::SampleCountFlagBits::e1,
        vk::ImageTiling::eOptimal,                  // GPU-optimal layout (not host-readable)
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::SharingMode::eExclusive
    );

    depthImage = device.createImage(imageInfo);

    vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(depthImage);
    // Find device-local memory for depth image
    auto memProps = physicalDevice.getMemoryProperties();
    uint32_t memTypeIndex = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i))
            && (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal)) {
            memTypeIndex = i;
            break;
        }
    }
    depthImageMemory = device.allocateMemory(vk::MemoryAllocateInfo(memReqs.size, memTypeIndex));
    device.bindImageMemory(depthImage, depthImageMemory, 0);

    // Image view — tells the pipeline how to interpret this image
    vk::ImageViewCreateInfo viewInfo(
        {},
        depthImage,
        vk::ImageViewType::e2D,
        depthFormat,
        {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1)
    );

    depthImageView = device.createImageView(viewInfo);
    std::cout << "[KMRB] Depth buffer created (" << extent.width << "x" << extent.height << ")" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FRAMEBUFFERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createFramebuffers(vk::Device device,
                                  const std::vector<vk::ImageView>& swapchainImageViews) {
    framebuffers.resize(swapchainImageViews.size());

    for (size_t i = 0; i < swapchainImageViews.size(); i++) {
        // Color + depth attachments (must match render pass attachment order)
        vk::ImageView attachments[] = { swapchainImageViews[i], depthImageView };

        vk::FramebufferCreateInfo framebufferInfo(
            {}, renderPass, 2, attachments,
            extent.width, extent.height, 1
        );

        framebuffers[i] = device.createFramebuffer(framebufferInfo);
    }

    std::cout << "[KMRB] Framebuffers created (" << framebuffers.size() << ")" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GLOBAL UBO BUFFERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createGlobalUBOBuffers(vk::Device device) {
    for (uint32_t i = 0; i < imageCount; i++) {
        std::string name = "global_ubo_" + std::to_string(i);
        bufferManager.createBuffer(name, sizeof(GlobalUBO),
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            true); // Persistent map
    }

    std::cout << "[KMRB] Global UBO buffers created (" << imageCount << ")" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DESCRIPTOR POOL & SETS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createDescriptorPool(vk::Device device) {
    // Pool sizes — room to grow. V2 will add samplers, storage buffers, etc.
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eUniformBuffer, imageCount },           // Global UBOs
        { vk::DescriptorType::eStorageBuffer, 8 },                    // Double-buffered SSBOs
        { vk::DescriptorType::eCombinedImageSampler, imageCount * 8 } // V2: textures
    };

    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        imageCount * 10,    // Max total sets (generous for V2)
        static_cast<uint32_t>(poolSizes.size()), poolSizes.data()
    );

    descriptorPool = device.createDescriptorPool(poolInfo);
    std::cout << "[KMRB] Descriptor pool created" << std::endl;
}

void Renderer::createDescriptorSets(vk::Device device) {
    // Allocate one global descriptor set per swapchain image
    std::vector<vk::DescriptorSetLayout> layouts(imageCount, descriptorSetLayouts[DESCRIPTOR_SET_GLOBAL]);

    vk::DescriptorSetAllocateInfo allocInfo(
        descriptorPool,
        static_cast<uint32_t>(layouts.size()),
        layouts.data()
    );

    globalDescriptorSets = device.allocateDescriptorSets(allocInfo);

    for (uint32_t i = 0; i < imageCount; i++) {
        std::string name = "global_ubo_" + std::to_string(i);
        vk::DescriptorBufferInfo bufferInfo(bufferManager.getBuffer(name), 0, sizeof(GlobalUBO));

        vk::WriteDescriptorSet descriptorWrite(
            globalDescriptorSets[i],
            0,                                     // Binding 0
            0,                                     // Array element
            1,                                     // Descriptor count
            vk::DescriptorType::eUniformBuffer,
            nullptr,                               // Image info (not used)
            &bufferInfo
        );

        device.updateDescriptorSets(descriptorWrite, nullptr);
    }

    // Allocate two particle descriptor sets (Set 2) — one for each ping-pong direction
    std::array<vk::DescriptorSetLayout, 2> particleLayouts = {
        descriptorSetLayouts[DESCRIPTOR_SET_OBJECT],
        descriptorSetLayouts[DESCRIPTOR_SET_OBJECT]
    };
    vk::DescriptorSetAllocateInfo particleAllocInfo(
        descriptorPool, static_cast<uint32_t>(particleLayouts.size()), particleLayouts.data()
    );
    auto particleSets = device.allocateDescriptorSets(particleAllocInfo);
    particleDescriptorSets[0] = particleSets[0];
    particleDescriptorSets[1] = particleSets[1];

    std::cout << "[KMRB] Descriptor sets allocated (" << imageCount << " global + 2 particle)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PARTICLE SSBO
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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

    // Wire up descriptor sets for ping-pong
    for (int i = 0; i < 2; i++) {
        vk::Buffer bufA = bufferManager.getBuffer(names[i]);
        vk::Buffer bufB = bufferManager.getBuffer(names[1 - i]);
        vk::DescriptorBufferInfo inputInfo(bufA, 0, bufferSize);
        vk::DescriptorBufferInfo outputInfo(bufB, 0, bufferSize);

        std::array<vk::WriteDescriptorSet, 2> writes = {{
            { particleDescriptorSets[i], 0, 0, 1,
              vk::DescriptorType::eStorageBuffer, nullptr, &inputInfo },
            { particleDescriptorSets[i], 1, 0, 1,
              vk::DescriptorType::eStorageBuffer, nullptr, &outputInfo }
        }};

        device.updateDescriptorSets(writes, nullptr);
    }

    std::cout << "[KMRB] Particle SSBOs allocated (2x " << particleCount << ", double-buffered)" << std::endl;
}

void Renderer::uploadParticles(vk::Device device, const std::vector<Particle>& particles) {
    particleCount = static_cast<uint32_t>(particles.size());
    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;

    bufferManager.upload("particle_a", particles.data(), bufferSize);
    bufferManager.upload("particle_b", particles.data(), bufferSize);

    std::cout << "[KMRB] Particles uploaded (" << particleCount << " from ECS)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMAND POOL & BUFFERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createCommandPool(vk::Device device, uint32_t graphicsQueueFamily) {
    vk::CommandPoolCreateInfo poolInfo(
        vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        graphicsQueueFamily
    );

    commandPool = device.createCommandPool(poolInfo);
    std::cout << "[KMRB] Command pool created" << std::endl;
}

void Renderer::createCommandBuffers(vk::Device device) {
    vk::CommandBufferAllocateInfo allocInfo(
        commandPool, vk::CommandBufferLevel::ePrimary, imageCount
    );

    commandBuffers = device.allocateCommandBuffers(allocInfo);
    std::cout << "[KMRB] Command buffers allocated (" << commandBuffers.size() << ")" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SYNC OBJECTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createSyncObjects(vk::Device device) {
    acquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphores.resize(imageCount);

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        acquireSemaphores[i] = device.createSemaphore(semaphoreInfo);
        inFlightFences[i] = device.createFence(fenceInfo);
    }
    for (uint32_t i = 0; i < imageCount; i++) {
        renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
    }

    imagesInFlight.assign(imageCount, nullptr);
    std::cout << "[KMRB] Sync objects created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// UBO UPDATE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::updateGlobalUBO(uint32_t imageIndex) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float>(currentTime - startTime).count();

    GlobalUBO ubo{};
    ubo.view = glm::lookAt(
        glm::vec3(0.0f, 1.0f, 4.0f),  // Camera position — pulled back to see the sphere
        glm::vec3(0.0f, 0.0f, 0.0f),  // Look at origin
        glm::vec3(0.0f, 1.0f, 0.0f)   // Up vector
    );
    ubo.proj = glm::perspective(
        glm::radians(45.0f),
        static_cast<float>(extent.width) / static_cast<float>(extent.height),
        0.1f, 100.0f
    );
    // GLM was designed for OpenGL where Y clip coord is inverted vs Vulkan
    ubo.proj[1][1] *= -1;

    ubo.cameraPos = glm::vec4(0.0f, 1.0f, 4.0f, 1.0f);
    ubo.deltaTime = time - elapsedTime;
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

    // ── COMPUTE PASS: update particle positions ──
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, computePipeline);

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, pipelineLayout,
        DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr
    );

    // Bind the current ping-pong set: reads buffer [pingPong], writes buffer [1-pingPong]
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eCompute, pipelineLayout,
        DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr
    );

    uint32_t groupCount = (particleCount + 255) / 256;
    cmd.dispatch(groupCount, 1, 1);

    // Barrier: compute writes to output buffer must finish before vertex shader reads it
    vk::Buffer outputBuffer = bufferManager.getBuffer(pingPong == 0 ? "particle_b" : "particle_a");
    vk::BufferMemoryBarrier barrier(
        vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eShaderRead,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        outputBuffer, 0, VK_WHOLE_SIZE
    );

    cmd.pipelineBarrier(
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eVertexShader,
        {}, nullptr, barrier, nullptr
    );

    // ── RENDER PASS: draw particles ──
    std::array<vk::ClearValue, 2> clearValues = {
        vk::ClearValue(vk::ClearColorValue(std::array<float,4>{0.04f, 0.04f, 0.06f, 1.0f})),
        vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))
    };

    vk::RenderPassBeginInfo renderPassBeginInfo(
        renderPass, framebuffers[imageIndex],
        vk::Rect2D({0, 0}, extent),
        static_cast<uint32_t>(clearValues.size()), clearValues.data()
    );

    cmd.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    vk::Viewport viewport(0.0f, 0.0f,
        static_cast<float>(extent.width), static_cast<float>(extent.height),
        0.0f, 1.0f);
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D({0, 0}, extent));

    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipelineLayout,
        DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr
    );
    // Same ping-pong set — vertex shader reads binding 1 (the output compute just wrote)
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipelineLayout,
        DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr
    );

    PushConstants push{};
    push.model = glm::mat4(1.0f);
    push.color = glm::vec4(1.0f);
    cmd.pushConstants(pipelineLayout,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
        0, sizeof(PushConstants), &push);

    cmd.draw(particleCount, 1, 0, 0);

    cmd.endRenderPass();
    cmd.end();

    // Swap ping-pong — next frame reads what this frame wrote
    pingPong = 1 - pingPong;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DRAW FRAME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool Renderer::drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                         vk::Queue graphicsQueue, vk::Queue presentQueue) {
    // Poll for shader changes every ~0.5s using a simple frame counter
    static uint32_t frameCounter = 0;
    if (++frameCounter >= 30) {  // ~0.5s at 60fps
        frameCounter = 0;
        checkShaderReload(device);
    }

    (void)device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

    // Acquire — may throw OUT_OF_DATE if window was resized
    uint32_t imageIndex;
    try {
        auto [result, idx] = device.acquireNextImageKHR(
            swapchain, UINT64_MAX, acquireSemaphores[currentFrame], nullptr
        );
        imageIndex = idx;
        if (result == vk::Result::eSuboptimalKHR) {
            // Still usable this frame, but recreate after
        }
    } catch (vk::OutOfDateKHRError&) {
        return true; // Swapchain is stale, tell Core to recreate
    }

    if (imagesInFlight[imageIndex]) {
        (void)device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
    }
    imagesInFlight[imageIndex] = inFlightFences[currentFrame];

    device.resetFences(inFlightFences[currentFrame]);

    updateGlobalUBO(imageIndex);

    commandBuffers[imageIndex].reset();
    recordCommandBuffer(commandBuffers[imageIndex], imageIndex);

    vk::Semaphore waitSemaphores[] = { acquireSemaphores[currentFrame] };
    vk::Semaphore signalSemaphores[] = { renderFinishedSemaphores[imageIndex] };
    vk::PipelineStageFlags waitStages[] = { vk::PipelineStageFlagBits::eColorAttachmentOutput };

    vk::SubmitInfo submitInfo(
        1, waitSemaphores, waitStages,
        1, &commandBuffers[imageIndex],
        1, signalSemaphores
    );

    graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

    vk::PresentInfoKHR presentInfo(
        1, signalSemaphores,
        1, &swapchain, &imageIndex
    );

    bool needsRecreate = false;
    try {
        vk::Result presentResult = presentQueue.presentKHR(presentInfo);
        if (presentResult == vk::Result::eSuboptimalKHR) {
            needsRecreate = true;
        }
    } catch (vk::OutOfDateKHRError&) {
        needsRecreate = true;
    }

    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    return needsRecreate;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SWAPCHAIN RECREATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Destroy everything that depends on swapchain image count or extent
void Renderer::cleanupSwapchainResources(vk::Device device) {
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        device.destroySemaphore(acquireSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
    for (uint32_t i = 0; i < imageCount; i++) {
        device.destroySemaphore(renderFinishedSemaphores[i]);
    }
    device.destroyCommandPool(commandPool);
    for (auto& fb : framebuffers) {
        device.destroyFramebuffer(fb);
    }
    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);

    // Destroy UBO buffers (size-dependent, one per swapchain image)
    for (uint32_t i = 0; i < imageCount; i++) {
        bufferManager.destroyBuffer("global_ubo_" + std::to_string(i));
    }
    // Particle buffers survive — they're not size-dependent
    device.destroyDescriptorPool(descriptorPool);
}

// Recreate everything after swapchain was rebuilt with new size
void Renderer::onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                                   const std::vector<vk::ImageView>& newImageViews) {
    cleanupSwapchainResources(device);

    extent = newExtent;
    imageCount = static_cast<uint32_t>(newImageViews.size());
    currentFrame = 0;
    pingPong = 0;

    // Pipeline and render pass survive — only recreate size-dependent resources
    createDepthResources(device);
    createFramebuffers(device, newImageViews);
    createGlobalUBOBuffers(device);
    createDescriptorPool(device);
    createDescriptorSets(device);

    // Re-wire particle buffers to the new descriptor sets (buffers survive, sets were reallocated)
    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;
    const char* names[] = { "particle_a", "particle_b" };
    for (int i = 0; i < 2; i++) {
        vk::DescriptorBufferInfo inputInfo(bufferManager.getBuffer(names[i]), 0, bufferSize);
        vk::DescriptorBufferInfo outputInfo(bufferManager.getBuffer(names[1 - i]), 0, bufferSize);

        std::array<vk::WriteDescriptorSet, 2> writes = {{
            { particleDescriptorSets[i], 0, 0, 1,
              vk::DescriptorType::eStorageBuffer, nullptr, &inputInfo },
            { particleDescriptorSets[i], 1, 0, 1,
              vk::DescriptorType::eStorageBuffer, nullptr, &outputInfo }
        }};

        device.updateDescriptorSets(writes, nullptr);
    }

    createCommandPool(device, graphicsQueueFamily);
    createCommandBuffers(device);
    createSyncObjects(device);

    std::cout << "[KMRB] Renderer recreated for " << newExtent.width << "x" << newExtent.height << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SHADER HOT-RELOAD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::setComputeShader(const std::string& path) {
    computeShaderPath = path;
    if (std::filesystem::exists(path)) {
        lastShaderModTime = std::filesystem::last_write_time(path);
    }
}

// Called each frame — polls file modification time, recompiles + swaps pipeline on change
void Renderer::checkShaderReload(vk::Device device) {
    if (computeShaderPath.empty()) return;
    if (!std::filesystem::exists(computeShaderPath)) return;

    auto modTime = std::filesystem::last_write_time(computeShaderPath);
    if (modTime == lastShaderModTime) return;

    lastShaderModTime = modTime;
    std::cout << "[KMRB] Shader change detected: " << computeShaderPath << std::endl;

    auto spirv = compileGLSL(computeShaderPath);

    if (spirv.empty()) {
        std::cerr << "[KMRB] Hot-reload FAILED — keeping old pipeline" << std::endl;
        return;
    }

    // Wait for GPU to finish using the old pipeline
    device.waitIdle();

    // Destroy old compute pipeline
    device.destroyPipeline(computePipeline);

    // Create new one
    vk::ShaderModuleCreateInfo moduleInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    vk::ShaderModule compModule = device.createShaderModule(moduleInfo);

    vk::PipelineShaderStageCreateInfo stageInfo(
        {}, vk::ShaderStageFlagBits::eCompute, compModule, "main"
    );
    vk::ComputePipelineCreateInfo pipelineInfo({}, stageInfo, pipelineLayout);

    auto result = device.createComputePipeline(nullptr, pipelineInfo);
    computePipeline = result.value;

    device.destroyShaderModule(compModule);
    std::cout << "[KMRB] Compute pipeline hot-reloaded!" << std::endl;
}

// Runtime GLSL → SPIR-V via glslc subprocess
std::vector<uint32_t> Renderer::compileGLSL(const std::string& sourcePath) {
    std::string outputPath = sourcePath + ".tmp.spv";

    // Compile with glslc — target Vulkan 1.3, optimize for performance
    std::string cmd = "glslc --target-env=vulkan1.3 -O \"" + sourcePath + "\" -o \"" + outputPath + "\" 2>&1";
    FILE* pipe = _popen(cmd.c_str(), "r");
    if (!pipe) {
        std::cerr << "[KMRB] Failed to run glslc" << std::endl;
        return {};
    }

    // Capture compiler output (errors/warnings)
    std::string compilerOutput;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe)) {
        compilerOutput += buf;
    }
    int exitCode = _pclose(pipe);

    if (exitCode != 0) {
        std::cerr << "[KMRB] Shader compilation error:\n" << compilerOutput << std::endl;
        std::filesystem::remove(outputPath);
        return {};
    }

    // Read the compiled SPIR-V
    std::ifstream file(outputPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[KMRB] Failed to read compiled SPIR-V" << std::endl;
        return {};
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(spirv.data()), fileSize);
    file.close();

    // Clean up temp file
    std::filesystem::remove(outputPath);

    if (!compilerOutput.empty()) {
        std::cout << "[KMRB] Shader warnings:\n" << compilerOutput;
    }

    return spirv;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VULKAN HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vk::ShaderModule Renderer::createShaderModule(vk::Device device, const std::vector<char>& code) {
    return device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, code.size(), reinterpret_cast<const uint32_t*>(code.data()))
    );
}

std::vector<char> Renderer::readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("KMRB: Failed to open file: " + filename);
    }
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
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        device.destroySemaphore(acquireSemaphores[i]);
        device.destroyFence(inFlightFences[i]);
    }
    for (uint32_t i = 0; i < imageCount; i++) {
        device.destroySemaphore(renderFinishedSemaphores[i]);
    }
    device.destroyCommandPool(commandPool);
    for (auto& fb : framebuffers) {
        device.destroyFramebuffer(fb);
    }
    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);
    bufferManager.cleanup(); // Destroys all managed buffers (UBOs, SSBOs)
    device.destroyDescriptorPool(descriptorPool);
    for (auto& layout : descriptorSetLayouts) {
        device.destroyDescriptorSetLayout(layout);
    }
    device.destroyPipeline(computePipeline);
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);
}

} // namespace kmrb
