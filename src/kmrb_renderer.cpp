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
                    uint32_t graphicsQueueFamily) {
    physicalDevice = gpu;
    extent = swapExtent;
    imageCount = static_cast<uint32_t>(swapchainImageViews.size());
    this->graphicsQueueFamily = graphicsQueueFamily;

    RenderPassConfig config{};
    config.colorFormat = swapchainFormat;
    config.hasDepth = true;
    config.depthFormat = depthFormat;

    createRenderPass(device, config);
    createDescriptorSetLayouts(device);
    createGraphicsPipeline(device);
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
        0,                                     // Binding index
        vk::DescriptorType::eUniformBuffer,
        1,                                     // Descriptor count
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
    );

    vk::DescriptorSetLayoutCreateInfo globalLayoutInfo({}, 1, &globalUBOBinding);
    descriptorSetLayouts[DESCRIPTOR_SET_GLOBAL] = device.createDescriptorSetLayout(globalLayoutInfo);

    // Set 1: Material — empty for V1
    vk::DescriptorSetLayoutCreateInfo materialLayoutInfo({}, 0, nullptr);
    descriptorSetLayouts[DESCRIPTOR_SET_MATERIAL] = device.createDescriptorSetLayout(materialLayoutInfo);

    // Set 2: Per-object — particle SSBO at binding 0
    vk::DescriptorSetLayoutBinding particleSSBOBinding(
        0,
        vk::DescriptorType::eStorageBuffer,
        1,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eCompute  // Vertex reads, compute writes
    );

    vk::DescriptorSetLayoutCreateInfo objectLayoutInfo({}, 1, &particleSSBOBinding);
    descriptorSetLayouts[DESCRIPTOR_SET_OBJECT] = device.createDescriptorSetLayout(objectLayoutInfo);

    std::cout << "[KMRB] Descriptor set layouts created (3 sets reserved)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GRAPHICS PIPELINE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createGraphicsPipeline(vk::Device device) {
    auto vertCode = readFile("shaders/particle.vert.spv");
    auto fragCode = readFile("shaders/particle.frag.spv");

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

    // Push constant range — 128 bytes for vertex+fragment stages
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
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

    // Allocate GPU-local memory (device-only, no CPU access needed)
    vk::MemoryRequirements memReqs = device.getImageMemoryRequirements(depthImage);
    depthImageMemory = device.allocateMemory(vk::MemoryAllocateInfo(
        memReqs.size,
        findMemoryType(memReqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)
    ));
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
    vk::DeviceSize bufferSize = sizeof(GlobalUBO);

    globalUBOBuffers.resize(imageCount);
    globalUBOMemory.resize(imageCount);
    globalUBOMapped.resize(imageCount);

    for (uint32_t i = 0; i < imageCount; i++) {
        createBuffer(device, bufferSize,
            vk::BufferUsageFlagBits::eUniformBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
            globalUBOBuffers[i], globalUBOMemory[i]);

        // Persistent mapping — keep the pointer for the lifetime of the buffer
        globalUBOMapped[i] = device.mapMemory(globalUBOMemory[i], 0, bufferSize);
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
        { vk::DescriptorType::eStorageBuffer, 4 },                    // Particle SSBOs
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

    // Point each descriptor set at its corresponding UBO buffer
    for (uint32_t i = 0; i < imageCount; i++) {
        vk::DescriptorBufferInfo bufferInfo(globalUBOBuffers[i], 0, sizeof(GlobalUBO));

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

    // Allocate particle descriptor set (Set 2) — written in createParticleBuffer
    vk::DescriptorSetLayout particleLayout = descriptorSetLayouts[DESCRIPTOR_SET_OBJECT];
    vk::DescriptorSetAllocateInfo particleAllocInfo(descriptorPool, 1, &particleLayout);
    particleDescriptorSet = device.allocateDescriptorSets(particleAllocInfo)[0];

    std::cout << "[KMRB] Descriptor sets allocated (" << imageCount << " global + 1 particle)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PARTICLE SSBO
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createParticleBuffer(vk::Device device) {
    // Generate particles in a sphere
    particleCount = 10000;
    std::vector<Particle> particles(particleCount);

    srand(42);
    for (uint32_t i = 0; i < particleCount; i++) {
        // Random point in a unit sphere
        float theta = static_cast<float>(rand()) / RAND_MAX * 6.2831853f;
        float phi = acos(1.0f - 2.0f * static_cast<float>(rand()) / RAND_MAX);
        float r = cbrt(static_cast<float>(rand()) / RAND_MAX) * 1.5f;

        float x = r * sin(phi) * cos(theta);
        float y = r * sin(phi) * sin(theta);
        float z = r * cos(phi);

        particles[i].position = glm::vec4(x, y, z, 2.0f);          // w = point size
        particles[i].velocity = glm::vec4(0.0f);
        particles[i].color = glm::vec4(
            0.4f + 0.6f * (r / 1.5f),   // Brighter at edges
            0.2f + 0.3f * (1.0f - r / 1.5f),
            0.8f,
            1.0f
        );
    }

    vk::DeviceSize bufferSize = sizeof(Particle) * particleCount;

    // STORAGE_BUFFER_BIT — this is what makes it an SSBO
    // Host-visible so CPU can upload. Later, compute shaders will write to device-local SSBOs.
    createBuffer(device, bufferSize,
        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
        particleBuffer, particleBufferMemory);

    // Upload particle data
    void* mapped = device.mapMemory(particleBufferMemory, 0, bufferSize);
    memcpy(mapped, particles.data(), bufferSize);
    device.unmapMemory(particleBufferMemory);

    // Point the descriptor set at this buffer
    vk::DescriptorBufferInfo ssboInfo(particleBuffer, 0, bufferSize);
    vk::WriteDescriptorSet write(
        particleDescriptorSet,
        0, 0, 1,
        vk::DescriptorType::eStorageBuffer,
        nullptr, &ssboInfo
    );
    device.updateDescriptorSets(write, nullptr);

    std::cout << "[KMRB] Particle SSBO created (" << particleCount << " particles)" << std::endl;
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

    memcpy(globalUBOMapped[imageIndex], &ubo, sizeof(ubo));
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// COMMAND RECORDING
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::recordCommandBuffer(vk::CommandBuffer cmd, uint32_t imageIndex) {
    cmd.begin(vk::CommandBufferBeginInfo{});

    // Clear values must match attachment order: [0] = color, [1] = depth
    std::array<vk::ClearValue, 2> clearValues = {
        vk::ClearValue(vk::ClearColorValue(std::array<float,4>{0.04f, 0.04f, 0.06f, 1.0f})),
        vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))  // 1.0 = max depth (far plane)
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

    // Bind Set 0 (global UBO)
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelineLayout,
        DESCRIPTOR_SET_GLOBAL,
        globalDescriptorSets[imageIndex],
        nullptr
    );

    // Bind Set 2 (particle SSBO)
    cmd.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics,
        pipelineLayout,
        DESCRIPTOR_SET_OBJECT,
        particleDescriptorSet,
        nullptr
    );

    // Push constants — slow spin so you can see the 3D sphere shape
    PushConstants push{};
    push.model = glm::rotate(glm::mat4(1.0f), elapsedTime * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));
    push.color = glm::vec4(1.0f);
    cmd.pushConstants(pipelineLayout,
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, sizeof(PushConstants), &push);

    // Draw one vertex per particle — shader reads position from SSBO via gl_VertexIndex
    cmd.draw(particleCount, 1, 0, 0);

    cmd.endRenderPass();
    cmd.end();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DRAW FRAME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool Renderer::drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                         vk::Queue graphicsQueue, vk::Queue presentQueue) {
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
    // Depth buffer depends on extent
    device.destroyImageView(depthImageView);
    device.destroyImage(depthImage);
    device.freeMemory(depthImageMemory);

    for (uint32_t i = 0; i < imageCount; i++) {
        device.unmapMemory(globalUBOMemory[i]);
        device.destroyBuffer(globalUBOBuffers[i]);
        device.freeMemory(globalUBOMemory[i]);
    }
    device.destroyDescriptorPool(descriptorPool);
}

// Recreate everything after swapchain was rebuilt with new size
void Renderer::onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                                   const std::vector<vk::ImageView>& newImageViews) {
    cleanupSwapchainResources(device);

    extent = newExtent;
    imageCount = static_cast<uint32_t>(newImageViews.size());
    currentFrame = 0;

    // Pipeline and render pass survive — only recreate size-dependent resources
    createDepthResources(device);
    createFramebuffers(device, newImageViews);
    createGlobalUBOBuffers(device);
    createDescriptorPool(device);
    createDescriptorSets(device);
    createCommandPool(device, graphicsQueueFamily);
    createCommandBuffers(device);
    createSyncObjects(device);

    std::cout << "[KMRB] Renderer recreated for " << newExtent.width << "x" << newExtent.height << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VULKAN HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createBuffer(vk::Device device, vk::DeviceSize size,
                            vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                            vk::Buffer& buffer, vk::DeviceMemory& memory) {
    buffer = device.createBuffer(vk::BufferCreateInfo({}, size, usage));

    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(buffer);
    memory = device.allocateMemory(vk::MemoryAllocateInfo(
        memReqs.size, findMemoryType(memReqs.memoryTypeBits, properties)
    ));
    device.bindBufferMemory(buffer, memory, 0);
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("KMRB: Failed to find suitable memory type!");
}

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
    for (uint32_t i = 0; i < imageCount; i++) {
        device.unmapMemory(globalUBOMemory[i]);
        device.destroyBuffer(globalUBOBuffers[i]);
        device.freeMemory(globalUBOMemory[i]);
    }
    device.destroyBuffer(particleBuffer);
    device.freeMemory(particleBufferMemory);
    device.destroyDescriptorPool(descriptorPool);
    for (auto& layout : descriptorSetLayouts) {
        device.destroyDescriptorSetLayout(layout);
    }
    device.destroyPipeline(graphicsPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(renderPass);
}

} // namespace kmrb
