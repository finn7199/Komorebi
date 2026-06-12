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
#include <bit>
#include <array>
#include <execution>
#include "spirv_reflect.h"
#include <stb_image.h>

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
    renderExtent = swapExtent;  // Default render res = window size, overridden by Preferences
    colorFormat = swapchainFormat;
    imageCount = static_cast<uint32_t>(swapchainImageViews.size());
    this->graphicsQueueFamily = graphicsQueueFamily;
    particleCount = initialParticleCount;

    bufferManager.init(device, gpu);
    camera.init(glm::vec3(0.0f, 2.0f, 5.0f), -15.0f, -90.0f);

    appStartTime = std::chrono::steady_clock::now();
    lastFrameTime = appStartTime;

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

    // Command pool must exist before device-local buffers — staging copies record into it
    createCommandPool(device, graphicsQueueFamily);
    bufferManager.setTransferContext(commandPool, graphicsQueue);

    createParticleBuffer(device);
    createCommandBuffers(device);
    createSyncObjects(device);

    // Environment map: create 1x1 black placeholder so Set 1 is always valid,
    // then build the skybox rendering pipeline
    createPlaceholderCubemap(device);
    createSkyboxPipeline(device);

    // IBL: create the precomputed-lighting images + bake pipelines, bake the
    // BRDF LUT (one-time), then bake irradiance/prefiltered from the placeholder
    // (black → zero ambient until an HDR is loaded)
    createIBLResources(device);
    bakeIBLMaps(device);

    // Load built-in mesh primitives (cube, sphere, plane) so mesh entities work immediately
    meshCache.loadPrimitives(bufferManager);

    ui.init(window, instance, gpu, device, graphicsQueueFamily,
            graphicsQueue, swapchainPass, imageCount);


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

    std::array<vk::SubpassDependency, 2> dependencies = {{
        // Entry: wait for any previous use of the attachments before we write them
        { VK_SUBPASS_EXTERNAL, 0,
          vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
          vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
          {},
          vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite },
        // Exit: color writes must finish before ImGui samples this image as a texture
        { 0, VK_SUBPASS_EXTERNAL,
          vk::PipelineStageFlagBits::eColorAttachmentOutput,
          vk::PipelineStageFlagBits::eFragmentShader,
          vk::AccessFlagBits::eColorAttachmentWrite,
          vk::AccessFlagBits::eShaderRead }
    }};

    vk::RenderPassCreateInfo info({},
        static_cast<uint32_t>(attachments.size()), attachments.data(),
        1, &subpass, static_cast<uint32_t>(dependencies.size()), dependencies.data());

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

    // Set 1: Environment — available to all fragment and compute shaders.
    //   binding 0: raw env cubemap   — layout(set=1, binding=0) uniform samplerCube envMap;
    //   binding 1: IBL irradiance    — diffuse ambient (declared in kmrb_lighting.glsl)
    //   binding 2: IBL prefiltered   — specular ambient, mip = roughness
    //   binding 3: BRDF LUT          — split-sum integration table
    auto envStages = vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute;
    std::array<vk::DescriptorSetLayoutBinding, 4> envBindings = {{
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, envStages },
        { 1, vk::DescriptorType::eCombinedImageSampler, 1, envStages },
        { 2, vk::DescriptorType::eCombinedImageSampler, 1, envStages },
        { 3, vk::DescriptorType::eCombinedImageSampler, 1, envStages }
    }};
    vk::DescriptorSetLayoutCreateInfo materialLayoutInfo(
        {}, static_cast<uint32_t>(envBindings.size()), envBindings.data());
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
    // 128 bytes = Vulkan-guaranteed minimum for push constants.
    // Engine built-ins (model mat4 + color vec4) occupy bytes 0-79.
    // Bytes 80-127 are available for user-defined shader parameters
    // discovered via SPIRV-Reflect and tweaked live in the Inspector.
    vk::PushConstantRange pushRange(
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
        0, 128);

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

    // Clamp to the GPU buffer's capacity (sized for up to 100 cells below)
    int n = std::clamp(props.cellCount, 1, 100);
    float half = props.size / 2.0f;
    float step = props.size / static_cast<float>(n);

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
// LIGHT GIZMOS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::appendCross(std::vector<glm::vec3>& out, const glm::vec3& c, float s) {
    out.push_back(c + glm::vec3(-s, 0, 0)); out.push_back(c + glm::vec3(s, 0, 0));
    out.push_back(c + glm::vec3(0, -s, 0)); out.push_back(c + glm::vec3(0, s, 0));
    out.push_back(c + glm::vec3(0, 0, -s)); out.push_back(c + glm::vec3(0, 0, s));
}

void Renderer::appendSphereRings(std::vector<glm::vec3>& out, const glm::vec3& c, float r, int seg) {
    const float TAU = 6.28318530718f;
    for (int i = 0; i < seg; ++i) {
        float a0 = TAU * i / seg, a1 = TAU * (i + 1) / seg;
        float c0 = cosf(a0), s0 = sinf(a0), c1 = cosf(a1), s1 = sinf(a1);
        out.push_back(c + glm::vec3(r*c0, r*s0, 0)); out.push_back(c + glm::vec3(r*c1, r*s1, 0));
        out.push_back(c + glm::vec3(r*c0, 0, r*s0)); out.push_back(c + glm::vec3(r*c1, 0, r*s1));
        out.push_back(c + glm::vec3(0, r*c0, r*s0)); out.push_back(c + glm::vec3(0, r*c1, r*s1));
    }
}

void Renderer::appendSpotCone(std::vector<glm::vec3>& out, const glm::vec3& apex,
                              const glm::vec3& dir, float angleDeg, float range, int seg) {
    const float TAU = 6.28318530718f;
    glm::vec3 d = glm::normalize(dir);
    glm::vec3 up  = (fabsf(d.y) > 0.99f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 rgt = glm::normalize(glm::cross(d, up));
    glm::vec3 u   = glm::cross(rgt, d);

    float ringR  = range * tanf(glm::radians(angleDeg));
    glm::vec3 center = apex + d * range;

    // Center aim line
    out.push_back(apex); out.push_back(center);

    glm::vec3 prev{};
    for (int i = 0; i <= seg; ++i) {
        float a = TAU * i / seg;
        glm::vec3 p = center + (rgt * cosf(a) + u * sinf(a)) * ringR;
        if (i > 0) { out.push_back(prev); out.push_back(p); }
        prev = p;
        if (i > 0 && i % (seg / 4) == 0) { out.push_back(apex); out.push_back(p); }
    }
}

void Renderer::appendDirArrows(std::vector<glm::vec3>& out, const glm::vec3& origin,
                               const glm::vec3& dir, float len) {
    glm::vec3 d = glm::normalize(dir);
    glm::vec3 up  = (fabsf(d.y) > 0.99f) ? glm::vec3(1, 0, 0) : glm::vec3(0, 1, 0);
    glm::vec3 rgt = glm::normalize(glm::cross(d, up));
    const float head = 0.25f;
    glm::vec3 a = origin, b = origin + d * len;
    out.push_back(a); out.push_back(b);
    out.push_back(b); out.push_back(b - d*head + rgt*head*0.5f);
    out.push_back(b); out.push_back(b - d*head - rgt*head*0.5f);
}

void Renderer::updateGizmoBuffer(vk::Device device) {
    gizmoDraws.clear();
    if (!registry || !ui.getShowGizmos()) return;

    // Fixed-capacity GPU buffer — never write more vertices than fit in it
    constexpr uint32_t maxGizmoVerts = 8 * 400;
    constexpr vk::DeviceSize maxBuf = sizeof(glm::vec3) * maxGizmoVerts;
    constexpr uint32_t worstCasePerLight = 160;  // Point light: sphere rings + cross ≈ 150 verts

    gizmoVerts.clear();  // Member scratch — keeps its capacity across frames
    entt::entity sel = ui.getSelectedEntity();

    registry->view<LightComponent, Transform>().each([&](auto entity, auto& lc, auto& lt) {
        uint32_t first = static_cast<uint32_t>(gizmoVerts.size());
        if (first + worstCasePerLight > maxGizmoVerts) return;  // Buffer full — skip extra lights

        float pitch = glm::radians(lt.rotation.x);
        float yaw   = glm::radians(lt.rotation.y);
        glm::vec3 dir(cosf(pitch) * sinf(yaw), -sinf(pitch), -cosf(pitch) * cosf(yaw));

        if (lc.type == LightType::Point) {
            appendSphereRings(gizmoVerts, lt.position, 0.3f, 24);
            appendCross(gizmoVerts, lt.position, 0.15f);
        } else if (lc.type == LightType::Spot) {
            appendSpotCone(gizmoVerts, lt.position, dir, lc.spotAngle, 2.0f, 24);
            appendCross(gizmoVerts, lt.position, 0.15f);
        } else {
            // Directional — anchor near camera so it's always visible
            glm::vec3 anchor = camera.position + camera.getForward() * 4.0f;
            appendDirArrows(gizmoVerts, anchor, dir, 1.5f);
            appendCross(gizmoVerts, anchor, 0.15f);
        }

        GizmoDrawCmd dc;
        dc.firstVertex = first;
        dc.vertexCount = static_cast<uint32_t>(gizmoVerts.size()) - first;
        dc.color = glm::vec4(lc.color, 1.0f);
        dc.selected = (entity == sel);
        gizmoDraws.push_back(dc);
    });

    if (gizmoVerts.empty()) return;

    if (!bufferManager.exists("light_gizmos")) {
        bufferManager.createBuffer("light_gizmos", maxBuf,
            vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    bufferManager.upload("light_gizmos", gizmoVerts.data(), sizeof(glm::vec3) * gizmoVerts.size());
}

void Renderer::drawLightGizmos(vk::CommandBuffer cmd, uint32_t imageIndex) {
    if (!ui.getShowGizmos() || gizmoDraws.empty() || !bufferManager.exists("light_gizmos"))
        return;

    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, gridPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
        DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);

    vk::Buffer buf = bufferManager.getBuffer("light_gizmos");
    vk::DeviceSize offset = 0;
    cmd.bindVertexBuffers(0, buf, offset);

    for (const auto& dc : gizmoDraws) {
        PushConstants push{};
        push.model = glm::mat4(1.0f);
        // Selected light uses the editor's golden accent color
        push.color = dc.selected
            ? glm::vec4(0.78f, 0.64f, 0.30f, 1.0f)
            : dc.color;
        cmd.pushConstants(pipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &push);
        cmd.draw(dc.vertexCount, 1, dc.firstVertex, 0);
    }
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

    // Color image — sampled by ImGui as a texture (uses renderExtent, not swapchain extent)
    vk::ImageCreateInfo colorInfo({}, vk::ImageType::e2D, colorFormat,
        vk::Extent3D(renderExtent.width, renderExtent.height, 1), 1, 1,
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
        vk::Extent3D(renderExtent.width, renderExtent.height, 1), 1, 1,
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
        {}, offscreenPass, 2, fbAttachments, renderExtent.width, renderExtent.height, 1));

    kmrb::Log::info("Offscreen framebuffer created (" + std::to_string(renderExtent.width) + "x" + std::to_string(renderExtent.height) + ")");
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
        { vk::DescriptorType::eCombinedImageSampler, imageCount * 8 },
        { vk::DescriptorType::eStorageImage, 8 }  // transient IBL bake targets
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

    // Device-local: compute reads/writes these every frame — keep them in VRAM,
    // not host memory that the GPU would have to reach over PCIe
    for (int i = 0; i < 2; i++) {
        bufferManager.createBuffer(names[i], bufferSize,
            vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
            vk::MemoryPropertyFlagBits::eDeviceLocal);
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
    float time = std::chrono::duration<float>(
        std::chrono::steady_clock::now() - appStartTime).count();

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

    // Gather lights from ECS into the UBO array so user shaders can read them
    ubo.lightCount = 0;
    if (registry) {
        registry->view<LightComponent, Transform>().each([&](auto, auto& lc, auto& lt) {
            if (ubo.lightCount >= static_cast<int>(MAX_LIGHTS)) return;
            GPULight& gpu = ubo.lights[ubo.lightCount];
            gpu.positionAndType = glm::vec4(lt.position, static_cast<float>(lc.type));
            float pitch = glm::radians(lt.rotation.x);
            float yaw   = glm::radians(lt.rotation.y);
            glm::vec3 dir(cos(pitch) * sin(yaw), -sin(pitch), -cos(pitch) * cos(yaw));
            gpu.directionAndAngle = glm::vec4(glm::normalize(dir), cos(glm::radians(lc.spotAngle)));
            gpu.colorAndIntensity = glm::vec4(lc.color, lc.intensity);
            gpu.params = glm::vec4(lc.radius, 0, 0, 0);
            ubo.lightCount++;
        });
    }

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
        registry->view<PipelineComponent, ShaderProgramComponent>().each([&](auto entity, auto&, auto&) {
            if (activeInstance) return;
            auto it = shaderInstances.find(static_cast<uint32_t>(entity));
            if (it != shaderInstances.end()) activeInstance = &it->second;
        });
    }

    // ── INIT SHADER (one-shot: runs once to set up initial particle positions) ──
    // Dispatches when a Pipeline is first created, after reset, or when the init shader changes.
    // Writes to the output SSBO (binding 1), then skips the compute shader this frame
    // so it doesn't overwrite the init data. The vertex shader reads binding 1 on the
    // same frame, so the init result is visible immediately.
    bool initRanThisFrame = false;
    if (activeInstance && activeInstance->initPipeline && activeInstance->initPending) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, activeInstance->initPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr);

        PushConstants push{};
        push.model = glm::mat4(1.0f);
        push.color = glm::vec4(1.0f);
        cmd.pushConstants(pipelineLayout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
            0, sizeof(PushConstants), &push);

        cmd.dispatch((particleCount + activeInstance->initLocalSizeX - 1) / activeInstance->initLocalSizeX, 1, 1);

        // Barrier: init write → vertex read (this frame) AND compute read (next frame's dispatch)
        vk::Buffer outputBuffer = bufferManager.getBuffer(pingPong == 0 ? "particle_b" : "particle_a");
        vk::BufferMemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, outputBuffer, 0, VK_WHOLE_SIZE);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eComputeShader,
            {}, nullptr, barrier, nullptr);

        activeInstance->initPending = false;
        initRanThisFrame = true;
    }

    // ── COMPUTE (skip when init just ran — it would overwrite init data with zeros) ──
    bool shouldDispatch = ui.shouldDispatchCompute() && !initRanThisFrame;
    if (shouldDispatch && activeInstance && activeInstance->computePipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, activeInstance->computePipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, pipelineLayout,
            DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr);

        // Push constants must be set before dispatch — compute shader reads them too
        if (activeInstance->pushConstantSize > 0) {
            glm::mat4 model(1.0f);
            glm::vec4 color(1.0f);
            memcpy(activeInstance->pushConstantData.data(), &model, 64);
            memcpy(activeInstance->pushConstantData.data() + 64, &color, 16);
            cmd.pushConstants(pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                0, activeInstance->pushConstantSize, activeInstance->pushConstantData.data());
        } else {
            PushConstants push{};
            push.model = glm::mat4(1.0f);
            push.color = glm::vec4(1.0f);
            cmd.pushConstants(pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                0, sizeof(PushConstants), &push);
        }

        cmd.dispatch((particleCount + activeInstance->computeLocalSizeX - 1) / activeInstance->computeLocalSizeX, 1, 1);

        // Barrier: compute write → vertex read (this frame) AND compute read (next frame's dispatch)
        vk::Buffer outputBuffer = bufferManager.getBuffer(pingPong == 0 ? "particle_b" : "particle_a");
        vk::BufferMemoryBarrier barrier(
            vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, outputBuffer, 0, VK_WHOLE_SIZE);
        cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
            vk::PipelineStageFlagBits::eVertexShader | vk::PipelineStageFlagBits::eComputeShader,
            {}, nullptr, barrier, nullptr);
    }

    // ── OFFSCREEN PASS ──
    std::array<vk::ClearValue, 2> offscreenClear = {
        vk::ClearValue(vk::ClearColorValue(std::array<float,4>{0.04f, 0.04f, 0.06f, 1.0f})),
        vk::ClearValue(vk::ClearDepthStencilValue(1.0f, 0))
    };

    cmd.beginRenderPass(vk::RenderPassBeginInfo(
        offscreenPass, offscreenFramebuffer,
        vk::Rect2D({0,0}, renderExtent),
        static_cast<uint32_t>(offscreenClear.size()), offscreenClear.data()),
        vk::SubpassContents::eInline);

    vk::Viewport viewport(0, 0, (float)renderExtent.width, (float)renderExtent.height, 0, 1);
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D({0,0}, renderExtent));

    // ── SKYBOX (renders at far plane depth, behind everything) ──
    if (envMapLoaded && skyboxPipeline && envDescriptorSet) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, skyboxPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_MATERIAL, envDescriptorSet, nullptr);
        cmd.draw(3, 1, 0, 0);  // Fullscreen triangle (3 vertices, no VBO)
    }

    // Draw particles if we have an active graphics pipeline
    if (activeInstance && activeInstance->graphicsPipeline) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, activeInstance->graphicsPipeline);

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
        // Bind environment cubemap (Set 1) so user fragment shaders can sample it
        if (envDescriptorSet) {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
                DESCRIPTOR_SET_MATERIAL, envDescriptorSet, nullptr);
        }
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_OBJECT, particleDescriptorSets[pingPong], nullptr);

        // Push constants: engine built-ins (model + color) at fixed offsets,
        // plus any user-defined params from reflection at offsets 80+.
        // The Inspector writes user params directly into pushConstantData.
        if (activeInstance->pushConstantSize > 0) {
            // Write engine built-ins into the live data buffer each frame
            glm::mat4 model(1.0f);
            glm::vec4 color(1.0f);
            memcpy(activeInstance->pushConstantData.data(), &model, 64);
            memcpy(activeInstance->pushConstantData.data() + 64, &color, 16);

            cmd.pushConstants(pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                0, activeInstance->pushConstantSize, activeInstance->pushConstantData.data());
        } else {
            // Fallback: no reflection data yet, use fixed struct
            PushConstants push{};
            push.model = glm::mat4(1.0f);
            push.color = glm::vec4(1.0f);
            cmd.pushConstants(pipelineLayout,
                vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                0, sizeof(PushConstants), &push);
        }

        cmd.draw(particleCount, 1, 0, 0);
    }

    // ── MESH ENTITIES ──
    if (registry) {
        registry->view<MeshRendererComponent, Transform>().each([&](auto entity, auto& meshComp, auto& meshTransform) {
            uint32_t key = static_cast<uint32_t>(entity);
            auto instIt = meshInstances.find(key);
            if (instIt == meshInstances.end() || !instIt->second.graphicsPipeline) return;
            if (!meshCache.exists(meshComp.meshCacheKey)) return;

            const auto& gpuMesh = meshCache.get(meshComp.meshCacheKey);
            auto& inst = instIt->second;

            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, inst.graphicsPipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
                DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);
            if (envDescriptorSet) {
                cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
                    DESCRIPTOR_SET_MATERIAL, envDescriptorSet, nullptr);
            }

            glm::mat4 model = glm::translate(glm::mat4(1.0f), meshTransform.position);
            model = glm::rotate(model, glm::radians(meshTransform.rotation.y), glm::vec3(0, 1, 0));
            model = glm::rotate(model, glm::radians(meshTransform.rotation.x), glm::vec3(1, 0, 0));
            model = glm::rotate(model, glm::radians(meshTransform.rotation.z), glm::vec3(0, 0, 1));
            model = glm::scale(model, meshTransform.scale);

            if (inst.pushConstantSize > 0) {
                memcpy(inst.pushConstantData.data(), &model, 64);
                memcpy(inst.pushConstantData.data() + 64, &meshComp.color, 16);
                cmd.pushConstants(pipelineLayout,
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                    0, inst.pushConstantSize, inst.pushConstantData.data());
            } else {
                PushConstants push{};
                push.model = model;
                push.color = meshComp.color;
                cmd.pushConstants(pipelineLayout,
                    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eCompute,
                    0, sizeof(PushConstants), &push);
            }

            vk::Buffer vb = bufferManager.getBuffer(gpuMesh.vertexBufferName);
            vk::DeviceSize offset = 0;
            cmd.bindVertexBuffers(0, vb, offset);
            cmd.bindIndexBuffer(bufferManager.getBuffer(gpuMesh.indexBufferName), 0, vk::IndexType::eUint32);
            cmd.drawIndexed(gpuMesh.indexCount, 1, 0, 0, 0);
        });
    }

    // ── GRID LINES ──
    if (gridVertexCount > 0 && bufferManager.exists("grid_lines") && registry) {
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, gridPipeline);
        cmd.setViewport(0, viewport);
        // Scissor must stay inside the offscreen framebuffer (renderExtent, not window extent)
        cmd.setScissor(0, vk::Rect2D({0,0}, renderExtent));

        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout,
            DESCRIPTOR_SET_GLOBAL, globalDescriptorSets[imageIndex], nullptr);

        // Get grid entity position and color for push constants
        bool gridRendered = false;
        registry->view<GridComponent, Transform>().each([&](auto, auto& gridProps, auto& gridTransform) {
            if (gridRendered) return;
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
            gridRendered = true;
        });
    }

    // ── LIGHT GIZMOS (independent of the grid — visible even with no grid entity) ──
    drawLightGizmos(cmd, imageIndex);

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

    // Swap ping-pong buffers when compute dispatched OR init ran.
    // Init writes to binding 1 (output). Flipping makes that buffer the input for
    // next frame's compute shader, so it reads the init data instead of zeros.
    if (shouldDispatch || initRanThisFrame) pingPong = 1 - pingPong;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DRAW FRAME
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

bool Renderer::drawFrame(vk::Device device, vk::SwapchainKHR swapchain,
                         vk::Queue graphicsQueue, vk::Queue presentQueue,
                         GLFWwindow* window) {
    // Real frame delta — used for camera movement so fly speed doesn't depend on FPS
    auto now = std::chrono::steady_clock::now();
    float frameDt = std::chrono::duration<float>(now - lastFrameTime).count();
    lastFrameTime = now;
    frameDt = std::min(frameDt, 0.1f);  // Clamp gaps from window drags / debugger pauses

    // Poll shader files for hot-reload every 30 frames (not every frame)
    if (++hotReloadCounter >= 30) {
        hotReloadCounter = 0;
        syncShaderInstances(device);
        syncMeshInstances(device);
        updateGridBuffer(device);
    }
    if (!firstSyncDone && registry) {
        syncShaderInstances(device);
        syncMeshInstances(device);
        updateGridBuffer(device);
        firstSyncDone = true;
    }
    updateGizmoBuffer(device);

    // Sync preferences → engine state
    camera.moveSpeed = ui.getCameraMoveSpeed();
    camera.lookSensitivity = ui.getCameraLookSensitivity();

    // Resolution change — recreate offscreen framebuffer at new size
    if (ui.isRenderResolutionDirty()) {
        device.waitIdle();
        // Release the old ImGui binding first — AddTexture without RemoveTexture
        // leaks a descriptor set on every resolution change
        ImGui_ImplVulkan_RemoveTexture(offscreenImGuiDescriptor);
        cleanupOffscreenResources(device);
        renderExtent = vk::Extent2D(ui.getRenderWidth(), ui.getRenderHeight());
        createOffscreenResources(device);
        offscreenImGuiDescriptor = ImGui_ImplVulkan_AddTexture(
            offscreenSampler, offscreenColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Log::ok("Render resolution: " + std::to_string(renderExtent.width) + "x" + std::to_string(renderExtent.height));
    }

    // Particle count change — reallocate SSBOs and re-run init shader
    if (ui.isParticleCountDirty()) {
        device.waitIdle();
        uint32_t newCount = static_cast<uint32_t>(ui.getParticleCount());
        vk::DeviceSize bufferSize = sizeof(Particle) * newCount;
        bufferManager.destroyBuffer("particle_a");
        bufferManager.destroyBuffer("particle_b");
        const char* names[] = { "particle_a", "particle_b" };
        for (int i = 0; i < 2; i++) {
            bufferManager.createBuffer(names[i], bufferSize,
                vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eVertexBuffer,
                vk::MemoryPropertyFlagBits::eDeviceLocal);
            bufferManager.setElementInfo(names[i], newCount, sizeof(Particle));
        }
        // Re-bind descriptor sets for new buffers
        for (int i = 0; i < 2; i++) {
            vk::DescriptorBufferInfo inputInfo(bufferManager.getBuffer(names[i]), 0, bufferSize);
            vk::DescriptorBufferInfo outputInfo(bufferManager.getBuffer(names[1 - i]), 0, bufferSize);
            std::array<vk::WriteDescriptorSet, 2> writes = {{
                { particleDescriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputInfo },
                { particleDescriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputInfo }
            }};
            device.updateDescriptorSets(writes, nullptr);
        }
        particleCount = newCount;
        pingPong = 0;
        requestInitDispatch();  // Re-run init shader to fill new buffer
        Log::ok("Particle count: " + std::to_string(newCount));
    }

    // Camera — two-way sync with active camera entity
    camera.viewportHovered = ui.isViewportHovered();
    camera.update(window, frameDt);

    if (registry) {
        bool camFound = false;
        registry->view<CameraComponent, Transform>().each([&](auto, auto& cam, auto& t) {
            if (camFound || !cam.active) return;
            if (camera.isUserControlling) {
                t.position = camera.position;
                t.rotation = glm::vec3(camera.pitch, camera.yaw, camera.roll);
            } else {
                camera.position = t.position;
                camera.pitch = t.rotation.x;
                camera.yaw = t.rotation.y;
                camera.roll = t.rotation.z;
            }
            camFound = true;
        });
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

// Destroy only extent-dependent resources — pools, UBOs, descriptor sets,
// command buffers, and sync objects don't depend on the window size
void Renderer::cleanupSwapchainResources(vk::Device device) {
    for (auto& fb : swapchainFramebuffers)
        device.destroyFramebuffer(fb);
    swapchainFramebuffers.clear();

    cleanupOffscreenResources(device);
}

void Renderer::onSwapchainRecreate(vk::Device device, vk::Extent2D newExtent,
                                   const std::vector<vk::ImageView>& newImageViews) {
    uint32_t newImageCount = static_cast<uint32_t>(newImageViews.size());

    // Release the old viewport texture binding (avoids an ImGui descriptor leak)
    ImGui_ImplVulkan_RemoveTexture(offscreenImGuiDescriptor);

    if (newImageCount == imageCount) {
        // Fast path (normal resize): only the framebuffers and offscreen images
        // depend on the extent. Everything else survives.
        cleanupSwapchainResources(device);
        extent = newExtent;
        createOffscreenResources(device);
        createSwapchainFramebuffers(device, newImageViews);
        imagesInFlight.assign(imageCount, nullptr);
    } else {
        // Rare path: image count changed — per-image resources must be rebuilt too
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(acquireSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }
        for (uint32_t i = 0; i < imageCount; i++)
            device.destroySemaphore(renderFinishedSemaphores[i]);

        device.destroyCommandPool(commandPool);
        cleanupSwapchainResources(device);

        for (uint32_t i = 0; i < imageCount; i++)
            bufferManager.destroyBuffer("global_ubo_" + std::to_string(i));

        device.destroyDescriptorPool(descriptorPool);
        envDescriptorSet = nullptr;  // Freed with the pool — must be reallocated

        extent = newExtent;
        imageCount = newImageCount;
        currentFrame = 0;

        createOffscreenResources(device);
        createSwapchainFramebuffers(device, newImageViews);
        createGlobalUBOBuffers(device);
        createDescriptorPool(device);
        createDescriptorSets(device);

        // Re-allocate env map descriptor set (old one was freed with the pool)
        if (envCubemapView && envSampler) {
            vk::DescriptorSetAllocateInfo envAllocInfo(descriptorPool, 1,
                &descriptorSetLayouts[DESCRIPTOR_SET_MATERIAL]);
            envDescriptorSet = device.allocateDescriptorSets(envAllocInfo)[0];
            vk::DescriptorImageInfo imgInfo(envSampler, envCubemapView,
                vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet write(envDescriptorSet, 0, 0, 1,
                vk::DescriptorType::eCombinedImageSampler, &imgInfo);
            device.updateDescriptorSets(write, nullptr);
            // IBL maps survive the pool reset (images are intact) — re-point bindings 1-3
            writeIBLDescriptors(device);
        }

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
        // The old command pool handle is stale — re-point staging copies at the new one
        bufferManager.setTransferContext(commandPool, device.getQueue(graphicsQueueFamily, 0));
        createCommandBuffers(device);
        createSyncObjects(device);

        ui.onSwapchainRecreate(imageCount);
    }

    // Re-bind the offscreen image as an ImGui texture
    offscreenImGuiDescriptor = ImGui_ImplVulkan_AddTexture(
        offscreenSampler, offscreenColorView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    kmrb::Log::info("Swapchain resources recreated ("
                    + std::to_string(extent.width) + "x" + std::to_string(extent.height) + ")");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SHADER INSTANCE MANAGEMENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Read local_size_x from compiled SPIR-V so dispatch counts always match the
// workgroup size the user declared in their GLSL (falls back to 256)
static uint32_t reflectWorkgroupSizeX(const std::vector<uint32_t>& spirv) {
    SpvReflectShaderModule module;
    if (spvReflectCreateShaderModule(spirv.size() * sizeof(uint32_t), spirv.data(), &module)
        != SPV_REFLECT_RESULT_SUCCESS) {
        return 256;
    }
    uint32_t sizeX = 256;
    if (module.entry_point_count > 0 && module.entry_points[0].local_size.x > 0) {
        sizeX = module.entry_points[0].local_size.x;
    }
    spvReflectDestroyShaderModule(&module);
    return sizeX;
}

vk::Pipeline Renderer::buildComputePipeline(vk::Device device, const std::vector<uint32_t>& spirv) {
    vk::ShaderModuleCreateInfo moduleInfo({}, spirv.size() * sizeof(uint32_t), spirv.data());
    vk::ShaderModule compModule = device.createShaderModule(moduleInfo);
    vk::PipelineShaderStageCreateInfo stageInfo({}, vk::ShaderStageFlagBits::eCompute, compModule, "main");
    auto result = device.createComputePipeline(nullptr,
        vk::ComputePipelineCreateInfo({}, stageInfo, pipelineLayout));
    device.destroyShaderModule(compModule);
    return result.value;
}

vk::Pipeline Renderer::buildGraphicsPipeline(vk::Device device,
                                              const std::vector<uint32_t>& vertSpirv,
                                              const std::vector<uint32_t>& fragSpirv) {
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
    return result.value;
}

void Renderer::destroyShaderInstance(vk::Device device, ShaderInstance& inst) {
    if (inst.initPipeline) { device.destroyPipeline(inst.initPipeline); inst.initPipeline = nullptr; }
    if (inst.computePipeline) { device.destroyPipeline(inst.computePipeline); inst.computePipeline = nullptr; }
    if (inst.graphicsPipeline) { device.destroyPipeline(inst.graphicsPipeline); inst.graphicsPipeline = nullptr; }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MESH INSTANCE MANAGEMENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

vk::Pipeline Renderer::buildMeshGraphicsPipeline(vk::Device device,
                                                 const std::vector<uint32_t>& vertSpirv,
                                                 const std::vector<uint32_t>& fragSpirv,
                                                 bool wireframe) {
    vk::ShaderModule vertModule = device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, vertSpirv.size() * sizeof(uint32_t), vertSpirv.data()));
    vk::ShaderModule fragModule = device.createShaderModule(
        vk::ShaderModuleCreateInfo({}, fragSpirv.size() * sizeof(uint32_t), fragSpirv.data()));

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragModule, "main" }
    };

    // Mesh vertex input: position (vec3), normal (vec3), UV (vec2)
    auto bindingDesc = MeshVertex::getBindingDescription();
    auto attrDescs = MeshVertex::getAttributeDescriptions();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo(
        {}, 1, &bindingDesc,
        static_cast<uint32_t>(attrDescs.size()), attrDescs.data());

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
    vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE,
        wireframe ? vk::PolygonMode::eLine : vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise,
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
    return result.value;
}

void Renderer::destroyMeshInstance(vk::Device device, MeshShaderInstance& inst) {
    if (inst.graphicsPipeline) { device.destroyPipeline(inst.graphicsPipeline); inst.graphicsPipeline = nullptr; }
}

void Renderer::syncMeshInstances(vk::Device device) {
    if (!registry) return;
    namespace fs = std::filesystem;

    // waitIdle once per sync pass (before the first rebuild), not once per mesh
    bool waitedIdle = false;

    auto view = registry->view<MeshRendererComponent>();
    for (auto entity : view) {
        auto& mesh = view.get<MeshRendererComponent>(entity);
        uint32_t key = static_cast<uint32_t>(entity);

        // Load mesh if path is set but not yet cached
        if (!mesh.meshPath.empty() && mesh.meshCacheKey.empty()) {
            mesh.meshCacheKey = meshCache.load(mesh.meshPath, bufferManager);
            if (mesh.meshCacheKey.empty()) mesh.meshPath.clear(); // Load failed
        }

        // Default to cube primitive if no mesh file specified
        if (mesh.meshCacheKey.empty()) {
            mesh.meshCacheKey = "__primitive_cube";
        }

        // Sync GPU stats to component so the UI Data Output panel can display them
        if (meshCache.exists(mesh.meshCacheKey)) {
            const auto& gpu = meshCache.get(mesh.meshCacheKey);
            mesh.vertexCount = gpu.vertexCount;
            mesh.indexCount  = gpu.indexCount;
            vk::DeviceSize bytes = 0;
            if (bufferManager.exists(gpu.vertexBufferName))
                bytes += bufferManager.getInfo(gpu.vertexBufferName).size;
            if (bufferManager.exists(gpu.indexBufferName))
                bytes += bufferManager.getInfo(gpu.indexBufferName).size;
            mesh.gpuBytes = static_cast<uint32_t>(bytes);
        }

        auto it = meshInstances.find(key);

        // Hot-reload: check file modification times
        if (it != meshInstances.end() && !mesh.shaderDirty) {
            auto& inst = it->second;
            auto checkMod = [](const std::string& path, fs::file_time_type& lastMod) -> bool {
                if (path.empty() || !fs::exists(path)) return false;
                auto mod = fs::last_write_time(path);
                if (mod != lastMod) { lastMod = mod; return true; }
                return false;
            };
            bool wireframeChanged = (inst.wireframe != mesh.wireframe);
            if (wireframeChanged ||
                checkMod(inst.vertexPath, inst.vertModTime) ||
                checkMod(inst.fragmentPath, inst.fragModTime)) {
                mesh.shaderDirty = true;
                Log::info("Mesh shader change detected, recompiling...");
            }
        }

        if (!mesh.shaderDirty) continue;

        // Pipelines may still be in use by in-flight frames — wait before destroying
        if (!waitedIdle) { device.waitIdle(); waitedIdle = true; }

        if (it != meshInstances.end()) {
            destroyMeshInstance(device, it->second);
        }

        MeshShaderInstance inst;
        inst.owner = entity;
        inst.wireframe = mesh.wireframe;

        // Use engine defaults if user hasn't assigned custom shaders.
        // Must use KMRB_SHADER_DIR (absolute) — relative paths break when cwd is build/.
        std::string vertPath = mesh.vertexShaderPath.empty()
            ? std::string(KMRB_SHADER_DIR) + "/engine/mesh_basic.vert"
            : mesh.vertexShaderPath;
        std::string fragPath = mesh.fragmentShaderPath.empty()
            ? std::string(KMRB_SHADER_DIR) + "/engine/mesh_unlit.frag"
            : mesh.fragmentShaderPath;

        inst.vertexPath = vertPath;
        inst.fragmentPath = fragPath;

        if (fs::exists(vertPath) && fs::exists(fragPath)) {
            // Compile each stage once — the frag SPIR-V is reused for reflection
            auto vertSpirv = compileGLSL(vertPath);
            auto fragSpirv = compileGLSL(fragPath);

            if (!vertSpirv.empty() && !fragSpirv.empty()) {
                // Reflect push constants from the fragment shader (more likely to have user params).
                // Reuse the ShaderInstance reflection logic — MeshShaderInstance has the same fields.
                ShaderInstance tempReflect;
                reflectPushConstants(fragSpirv, tempReflect);
                inst.pushConstantSize = tempReflect.pushConstantSize;
                inst.pushConstantData = std::move(tempReflect.pushConstantData);
                inst.reflectedParams = std::move(tempReflect.reflectedParams);

                inst.graphicsPipeline = buildMeshGraphicsPipeline(device, vertSpirv, fragSpirv, mesh.wireframe);
                Log::ok("Compiled mesh: " + fs::path(vertPath).filename().string()
                        + " + " + fs::path(fragPath).filename().string());
            } else {
                Log::error("Failed to compile mesh shaders");
            }
            inst.vertModTime = fs::last_write_time(vertPath);
            inst.fragModTime = fs::last_write_time(fragPath);
        }

        meshInstances[key] = std::move(inst);
        mesh.shaderDirty = false;
    }

    // Clean up orphaned instances
    std::vector<uint32_t> toRemove;
    for (auto& [key, inst] : meshInstances) {
        if (!registry->valid(inst.owner)) {
            if (!waitedIdle) { device.waitIdle(); waitedIdle = true; }
            destroyMeshInstance(device, inst);
            toRemove.push_back(key);
        }
    }
    for (auto key : toRemove) meshInstances.erase(key);
}

void Renderer::clearMeshCache(vk::Device device) {
    device.waitIdle();

    // Destroy all mesh shader instance pipelines
    for (auto& [key, inst] : meshInstances) destroyMeshInstance(device, inst);
    meshInstances.clear();

    // Destroy GPU vertex/index buffers and clear the cache map
    meshCache.cleanup(bufferManager);

    // Re-upload built-in primitives so the default cube fallback still works
    meshCache.loadPrimitives(bufferManager);

    // Reset all MeshRendererComponent state so syncMeshInstances reloads everything
    if (registry) {
        auto view = registry->view<MeshRendererComponent>();
        for (auto e : view) {
            auto& m = view.get<MeshRendererComponent>(e);
            m.meshCacheKey.clear();
            m.vertexCount = 0;
            m.indexCount  = 0;
            m.gpuBytes    = 0;
            m.shaderDirty = true;
        }
    }

    Log::info("Mesh cache cleared — meshes will reload on next sync");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SPIRV-REFLECT: Extract push constant members from compiled SPIR-V
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// After compiling a shader to SPIR-V, this function reads the binary to find
// all push constant block members. It skips engine built-ins (model mat4 at
// offset 0 and color vec4 at offset 64) and stores the rest as ReflectedParam
// entries. The Inspector then auto-generates UI widgets from these params,
// letting users tweak shader values live without editing GLSL.

void Renderer::reflectPushConstants(const std::vector<uint32_t>& spirv, ShaderInstance& inst) {
    // Create a SPIRV-Reflect module from the compiled SPIR-V bytecode
    SpvReflectShaderModule module;
    SpvReflectResult result = spvReflectCreateShaderModule(
        spirv.size() * sizeof(uint32_t), spirv.data(), &module);

    if (result != SPV_REFLECT_RESULT_SUCCESS) {
        Log::warn("SPIRV-Reflect: failed to parse module");
        return;
    }

    // Check if this shader declares any push constant blocks
    if (module.push_constant_block_count == 0) {
        spvReflectDestroyShaderModule(&module);
        return;
    }

    // Read the first push constant block (shaders typically have only one)
    const SpvReflectBlockVariable& block = module.push_constant_blocks[0];

    // Allocate the push constant data buffer (128 bytes = Vulkan guaranteed minimum)
    uint32_t totalSize = block.size;
    if (totalSize > 128) totalSize = 128; // Clamp to Vulkan-guaranteed minimum

    // Only re-allocate if this is the first reflection or size changed.
    // Preserve existing user-tweaked values if the block layout hasn't changed.
    if (inst.pushConstantSize != totalSize) {
        inst.pushConstantData.resize(totalSize, 0);
        inst.pushConstantSize = totalSize;

        // Pre-fill engine built-ins with sensible defaults:
        //   Offset  0-63: identity mat4 (model matrix)
        //   Offset 64-79: vec4(1.0) (color/tint)
        glm::mat4 identity(1.0f);
        glm::vec4 white(1.0f);
        if (totalSize >= 64) memcpy(inst.pushConstantData.data(), &identity, 64);
        if (totalSize >= 80) memcpy(inst.pushConstantData.data() + 64, &white, 16);
    }

    // Walk each member of the push constant block and classify its type.
    // Engine built-ins (model at offset 0, color at offset 64) are skipped —
    // only user-defined params (offset >= 80) become Inspector widgets.
    inst.reflectedParams.clear();
    for (uint32_t i = 0; i < block.member_count; i++) {
        const SpvReflectBlockVariable& member = block.members[i];

        // Skip engine built-ins: model (offset 0, 64 bytes) and color (offset 64, 16 bytes)
        if (member.offset < 80) continue;

        ReflectedParam param;
        param.name = member.name ? member.name : "unnamed";
        param.offset = member.offset;
        param.size = member.size;

        // Determine the parameter type from SPIRV-Reflect type flags.
        // type_description->type_flags tells us if it's float/int/vector/matrix,
        // and numeric traits give us the vector component count.
        SpvReflectTypeFlags flags = member.type_description->type_flags;
        uint32_t vecSize = member.numeric.vector.component_count;
        uint32_t cols = member.numeric.matrix.column_count;

        if (flags & SPV_REFLECT_TYPE_FLAG_MATRIX) {
            // mat4 (or other matrix types)
            param.type = ReflectedParam::Mat4;
        } else if (flags & SPV_REFLECT_TYPE_FLAG_FLOAT) {
            // float, vec2, vec3, vec4 — distinguished by vector component count
            if (flags & SPV_REFLECT_TYPE_FLAG_VECTOR) {
                if (vecSize == 2) param.type = ReflectedParam::Vec2;
                else if (vecSize == 3) param.type = ReflectedParam::Vec3;
                else if (vecSize == 4) param.type = ReflectedParam::Vec4;
                else param.type = ReflectedParam::Unknown;
            } else {
                param.type = ReflectedParam::Float;  // Scalar float
            }
        } else if (flags & SPV_REFLECT_TYPE_FLAG_INT) {
            param.type = ReflectedParam::Int;
        } else if (flags & SPV_REFLECT_TYPE_FLAG_BOOL) {
            param.type = ReflectedParam::Bool;
        } else {
            param.type = ReflectedParam::Unknown;
        }

        inst.reflectedParams.push_back(param);
    }

    if (!inst.reflectedParams.empty()) {
        Log::info("Reflected " + std::to_string(inst.reflectedParams.size()) + " push constant param(s)");
    }

    spvReflectDestroyShaderModule(&module);
}

void Renderer::syncShaderInstances(vk::Device device) {
    if (!registry) return;
    namespace fs = std::filesystem;

    // waitIdle once per sync pass (before the first rebuild), not once per shader
    bool waitedIdle = false;

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
            if (checkMod(inst.initPath, inst.initModTime) ||
                checkMod(inst.computePath, inst.compModTime) ||
                checkMod(inst.vertexPath, inst.vertModTime) ||
                checkMod(inst.fragmentPath, inst.fragModTime)) {
                shader.dirty = true;
                Log::info("Shader change detected, recompiling...");
            }
        }

        if (!shader.dirty) continue;

        // Pipelines may still be in use by in-flight frames — wait before destroying
        if (!waitedIdle) { device.waitIdle(); waitedIdle = true; }

        if (it != shaderInstances.end()) {
            destroyShaderInstance(device, it->second);
        }

        ShaderInstance inst;
        inst.owner = entity;
        inst.initPath = shader.initPath;
        inst.computePath = shader.computePath;
        inst.vertexPath = shader.vertexPath;
        inst.fragmentPath = shader.fragmentPath;

        // Each shader is compiled to SPIR-V once; the same binary feeds the
        // pipeline, push constant reflection, and workgroup size reflection.
        // Push constants: all stages share one layout (Vulkan requires this),
        // so reflecting compute (or vertex as fallback) once is enough.
        bool reflected = false;

        // Build init pipeline (one-shot compute shader for particle setup)
        if (!shader.initPath.empty() && fs::exists(shader.initPath)) {
            auto spirv = compileGLSL(shader.initPath);
            if (!spirv.empty()) {
                inst.initPipeline = buildComputePipeline(device, spirv);
                inst.initLocalSizeX = reflectWorkgroupSizeX(spirv);
                Log::ok("Compiled init: " + fs::path(shader.initPath).filename().string());
            } else {
                Log::error("Failed to compile: " + shader.initPath);
            }
            inst.initModTime = fs::last_write_time(shader.initPath);
            inst.initPending = true;  // Dispatch once on next frame
        }

        if (!shader.computePath.empty() && fs::exists(shader.computePath)) {
            auto spirv = compileGLSL(shader.computePath);
            if (!spirv.empty()) {
                if (!reflected) { reflectPushConstants(spirv, inst); reflected = true; }
                inst.computePipeline = buildComputePipeline(device, spirv);
                inst.computeLocalSizeX = reflectWorkgroupSizeX(spirv);
                Log::ok("Compiled compute: " + fs::path(shader.computePath).filename().string());
            } else {
                Log::error("Failed to compile: " + shader.computePath);
            }
            inst.compModTime = fs::last_write_time(shader.computePath);
        }

        if (!shader.vertexPath.empty() && !shader.fragmentPath.empty() &&
            fs::exists(shader.vertexPath) && fs::exists(shader.fragmentPath)) {
            auto vertSpirv = compileGLSL(shader.vertexPath);
            auto fragSpirv = compileGLSL(shader.fragmentPath);
            if (!vertSpirv.empty() && !fragSpirv.empty()) {
                if (!reflected) { reflectPushConstants(vertSpirv, inst); reflected = true; }
                inst.graphicsPipeline = buildGraphicsPipeline(device, vertSpirv, fragSpirv);
                Log::ok("Compiled graphics: " + fs::path(shader.vertexPath).filename().string()
                        + " + " + fs::path(shader.fragmentPath).filename().string());
            } else {
                Log::error("Failed to compile graphics shaders");
            }
            inst.vertModTime = fs::last_write_time(shader.vertexPath);
            inst.fragModTime = fs::last_write_time(shader.fragmentPath);
        }

        shaderInstances[key] = std::move(inst);
        shader.dirty = false;
    }

    // Clean up orphaned instances
    std::vector<uint32_t> toRemove;
    for (auto& [key, inst] : shaderInstances) {
        if (!registry->valid(inst.owner)) {
            if (!waitedIdle) { device.waitIdle(); waitedIdle = true; }
            destroyShaderInstance(device, inst);
            toRemove.push_back(key);
        }
    }
    for (auto key : toRemove) shaderInstances.erase(key);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

std::vector<uint32_t> Renderer::compileGLSL(const std::string& sourcePath) {
    std::string outputPath = sourcePath + ".tmp.spv";
    // -g preserves variable names (OpName) in SPIR-V so SPIRV-Reflect can read them.
    // Without it, -O strips names and reflected params show as "unnamed".
    std::string cmd = "glslc --target-env=vulkan1.3 -O -g";
    cmd += " -I \"" + std::string(KMRB_SHADER_DIR) + "/include\"";
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
// ENVIRONMENT MAP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Helper: create a VkImage cubemap, allocate memory, create view and sampler.
// Used by both the 1x1 placeholder and the real HDR environment map.
static void createCubemapResources(
    vk::Device device, vk::PhysicalDevice gpu,
    uint32_t faceSize, vk::Format format, const void* pixelData, vk::DeviceSize dataSize,
    vk::Image& outImage, vk::DeviceMemory& outMemory,
    vk::ImageView& outView, vk::Sampler& outSampler,
    vk::CommandPool cmdPool, vk::Queue queue)
{
    // Find device-local memory type
    auto findMem = [&](vk::MemoryRequirements reqs, vk::MemoryPropertyFlags props) -> uint32_t {
        auto memProps = gpu.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((reqs.memoryTypeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
                return i;
        }
        throw std::runtime_error("Failed to find suitable memory type for cubemap");
    };

    // Create the cubemap image (6 array layers, one per face)
    vk::ImageCreateInfo imgInfo(
        vk::ImageCreateFlagBits::eCubeCompatible,
        vk::ImageType::e2D, format,
        vk::Extent3D(faceSize, faceSize, 1),
        1, 6,  // 1 mip level, 6 faces
        vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
        vk::SharingMode::eExclusive);
    outImage = device.createImage(imgInfo);

    // Allocate and bind device-local memory
    auto reqs = device.getImageMemoryRequirements(outImage);
    outMemory = device.allocateMemory(
        vk::MemoryAllocateInfo(reqs.size, findMem(reqs, vk::MemoryPropertyFlagBits::eDeviceLocal)));
    device.bindImageMemory(outImage, outMemory, 0);

    // Create staging buffer with the pixel data
    vk::BufferCreateInfo stagingInfo({}, dataSize, vk::BufferUsageFlagBits::eTransferSrc);
    vk::Buffer stagingBuf = device.createBuffer(stagingInfo);
    auto stagingReqs = device.getBufferMemoryRequirements(stagingBuf);
    vk::DeviceMemory stagingMem = device.allocateMemory(vk::MemoryAllocateInfo(
        stagingReqs.size, findMem(stagingReqs,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)));
    device.bindBufferMemory(stagingBuf, stagingMem, 0);

    // Copy pixel data to staging buffer
    void* mapped = device.mapMemory(stagingMem, 0, dataSize);
    memcpy(mapped, pixelData, dataSize);
    device.unmapMemory(stagingMem);

    // Record a one-shot command buffer to transition layout and copy data
    vk::CommandBuffer cmd = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(cmdPool, vk::CommandBufferLevel::ePrimary, 1))[0];
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Transition: undefined → transfer dst
    vk::ImageMemoryBarrier toTransfer(
        {}, vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        outImage, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, toTransfer);

    // Copy staging buffer → cubemap faces
    std::vector<vk::BufferImageCopy> copies(6);
    vk::DeviceSize faceBytes = static_cast<vk::DeviceSize>(faceSize) * faceSize * 4 * sizeof(uint16_t);
    // For placeholder (R16G16B16A16), faceBytes is small. For real HDR, computed from format.
    // We use dataSize / 6 to be format-agnostic.
    vk::DeviceSize bytesPerFace = dataSize / 6;
    for (uint32_t face = 0; face < 6; face++) {
        copies[face] = vk::BufferImageCopy(
            face * bytesPerFace, 0, 0,
            vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, face, 1),
            vk::Offset3D(0, 0, 0), vk::Extent3D(faceSize, faceSize, 1));
    }
    cmd.copyBufferToImage(stagingBuf, outImage, vk::ImageLayout::eTransferDstOptimal, copies);

    // Transition: transfer dst → shader read
    vk::ImageMemoryBarrier toShader(
        vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        outImage, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
        vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, toShader);

    cmd.end();
    vk::SubmitInfo submit({}, {}, cmd);
    queue.submit(submit);
    queue.waitIdle();

    device.freeCommandBuffers(cmdPool, cmd);
    device.destroyBuffer(stagingBuf);
    device.freeMemory(stagingMem);

    // Create cubemap image view
    outView = device.createImageView(vk::ImageViewCreateInfo(
        {}, outImage, vk::ImageViewType::eCube, format, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)));

    // Create sampler with linear filtering
    vk::SamplerCreateInfo samplerInfo({},
        vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
        vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
        vk::SamplerAddressMode::eClampToEdge);
    outSampler = device.createSampler(samplerInfo);
}

void Renderer::createPlaceholderCubemap(vk::Device device) {
    // 1x1 black cubemap — ensures Set 1 is always valid even before an HDR is loaded.
    // Without this, shaders that reference envMap would crash on an unbound descriptor.
    const uint32_t faceSize = 1;
    // 6 faces x 1x1 pixel x 4 channels x 2 bytes (R16G16B16A16)
    std::vector<uint16_t> blackPixels(6 * 1 * 1 * 4, 0);

    vk::Queue queue = device.getQueue(graphicsQueueFamily, 0);
    createCubemapResources(device, physicalDevice, faceSize,
        vk::Format::eR16G16B16A16Sfloat, blackPixels.data(),
        blackPixels.size() * sizeof(uint16_t),
        envCubemap, envCubemapMemory, envCubemapView, envSampler,
        commandPool, queue);

    // Allocate the Set 1 descriptor set on first call only — on env-map clear
    // this function runs again and must reuse the existing set, not leak a new one
    if (!envDescriptorSet) {
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1,
            &descriptorSetLayouts[DESCRIPTOR_SET_MATERIAL]);
        envDescriptorSet = device.allocateDescriptorSets(allocInfo)[0];
    }

    vk::DescriptorImageInfo imgInfo(envSampler, envCubemapView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(envDescriptorSet, 0, 0, 1,
        vk::DescriptorType::eCombinedImageSampler, &imgInfo);
    device.updateDescriptorSets(write, nullptr);

    Log::info("Placeholder cubemap created (1x1 black)");
}

void Renderer::destroyEnvironmentMap(vk::Device device) {
    device.waitIdle();

    if (envCubemapView) { device.destroyImageView(envCubemapView); envCubemapView = nullptr; }
    if (envSampler) { device.destroySampler(envSampler); envSampler = nullptr; }
    if (envCubemap) { device.destroyImage(envCubemap); envCubemap = nullptr; }
    if (envCubemapMemory) { device.freeMemory(envCubemapMemory); envCubemapMemory = nullptr; }

    envMapLoaded = false;

    // Recreate the placeholder so Set 1 is always valid, then re-bake the IBL
    // maps from it — black source convolves to black, killing the ambient light
    createPlaceholderCubemap(device);
    bakeIBLMaps(device);
    Log::info("Environment map cleared");
}

void Renderer::loadEnvironmentMap(vk::Device device, const std::string& hdrPath) {
    // Load the equirectangular HDR image using stb_image
    int width, height, channels;
    float* hdrData = stbi_loadf(hdrPath.c_str(), &width, &height, &channels, 4);
    if (!hdrData) {
        Log::error("Failed to load HDR: " + hdrPath);
        return;
    }

    device.waitIdle();

    // Destroy previous cubemap (placeholder or old HDR)
    if (envCubemapView) device.destroyImageView(envCubemapView);
    if (envSampler) device.destroySampler(envSampler);
    if (envCubemap) device.destroyImage(envCubemap);
    if (envCubemapMemory) device.freeMemory(envCubemapMemory);
    envCubemapView = nullptr; envSampler = nullptr;
    envCubemap = nullptr; envCubemapMemory = nullptr;

    // Decide cubemap face size — use height of equirectangular image (2:1 aspect ratio).
    // Cap at 1024 for VRAM sanity.
    uint32_t faceSize = std::min(static_cast<uint32_t>(height), 1024u);

    // Convert equirectangular → 6 cubemap faces on CPU.
    // For each face pixel, compute the 3D direction, convert to equirect UV, sample.
    // Store as R16G16B16A16 (half-float) to save VRAM while preserving HDR range.
    uint32_t facePixels = faceSize * faceSize;
    std::vector<uint16_t> cubemapData(6 * facePixels * 4);

    // Half-float conversion helper (IEEE 754 binary16)
    auto floatToHalf = [](float f) -> uint16_t {
        uint32_t bits = std::bit_cast<uint32_t>(f);  // Type-pun safely (no aliasing UB)
        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = bits & 0x7FFFFF;
        if (exp <= 0) return static_cast<uint16_t>(sign); // Underflow → 0
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00); // Overflow → inf
        return static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
    };

    // Direction vectors for each cubemap face.
    // For face (u,v) in [0,1]², these define the 3D direction to sample.
    // face 0: +X, face 1: -X, face 2: +Y, face 3: -Y, face 4: +Z, face 5: -Z
    // Each face writes a disjoint range of cubemapData → safe to convert in parallel.
    std::array<uint32_t, 6> faces = { 0, 1, 2, 3, 4, 5 };
    std::for_each(std::execution::par, faces.begin(), faces.end(), [&](uint32_t face) {
        for (uint32_t y = 0; y < faceSize; y++) {
            for (uint32_t x = 0; x < faceSize; x++) {
                // Map pixel to [-1, 1] range
                float u = (static_cast<float>(x) + 0.5f) / faceSize * 2.0f - 1.0f;
                float v = (static_cast<float>(y) + 0.5f) / faceSize * 2.0f - 1.0f;

                // Compute 3D direction for this face and pixel
                float dx, dy, dz;
                switch (face) {
                    case 0: dx =  1; dy = -v; dz = -u; break;  // +X
                    case 1: dx = -1; dy = -v; dz =  u; break;  // -X
                    case 2: dx =  u; dy =  1; dz =  v; break;  // +Y
                    case 3: dx =  u; dy = -1; dz = -v; break;  // -Y
                    case 4: dx =  u; dy = -v; dz =  1; break;  // +Z
                    case 5: dx = -u; dy = -v; dz = -1; break;  // -Z
                    default: dx = dy = dz = 0; break;
                }

                // Normalize direction
                float len = std::sqrt(dx*dx + dy*dy + dz*dz);
                dx /= len; dy /= len; dz /= len;

                // Convert direction to equirectangular UV
                float theta = std::atan2(dz, dx);             // -pi to pi
                float phi = std::asin(std::clamp(dy, -1.0f, 1.0f));  // -pi/2 to pi/2
                float eqU = (theta / (2.0f * 3.14159265f)) + 0.5f;   // 0 to 1
                float eqV = (phi / 3.14159265f) + 0.5f;              // 0 to 1

                // Sample equirectangular image (nearest neighbor)
                float sx = eqU * (width - 1);
                float sy = (1.0f - eqV) * (height - 1);  // Flip V
                int ix = std::clamp(static_cast<int>(sx), 0, width - 1);
                int iy = std::clamp(static_cast<int>(sy), 0, height - 1);
                const float* pixel = hdrData + (iy * width + ix) * 4;

                // Write as half-float
                uint32_t dstIdx = (face * facePixels + y * faceSize + x) * 4;
                cubemapData[dstIdx + 0] = floatToHalf(pixel[0]);
                cubemapData[dstIdx + 1] = floatToHalf(pixel[1]);
                cubemapData[dstIdx + 2] = floatToHalf(pixel[2]);
                cubemapData[dstIdx + 3] = floatToHalf(1.0f);
            }
        }
    });

    stbi_image_free(hdrData);

    // Upload cubemap to GPU
    vk::Queue queue = device.getQueue(graphicsQueueFamily, 0);
    createCubemapResources(device, physicalDevice, faceSize,
        vk::Format::eR16G16B16A16Sfloat, cubemapData.data(),
        cubemapData.size() * sizeof(uint16_t),
        envCubemap, envCubemapMemory, envCubemapView, envSampler,
        commandPool, queue);

    // Update descriptor set to point to the new cubemap
    vk::DescriptorImageInfo imgInfo(envSampler, envCubemapView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::WriteDescriptorSet write(envDescriptorSet, 0, 0, 1,
        vk::DescriptorType::eCombinedImageSampler, &imgInfo);
    device.updateDescriptorSets(write, nullptr);

    // Re-convolve the IBL maps from the new environment so lighting matches the sky
    bakeIBLMaps(device);

    envMapLoaded = true;
    namespace fs = std::filesystem;
    Log::ok("Environment map loaded: " + fs::path(hdrPath).filename().string()
            + " (" + std::to_string(faceSize) + "x" + std::to_string(faceSize) + " per face, IBL baked)");
}

void Renderer::createSkyboxPipeline(vk::Device device) {
    // Compile skybox shaders from pre-built SPV files
    auto vertCode = readFile(KMRB_SHADER_SPV_DIR "/skybox.vert.spv");
    auto fragCode = readFile(KMRB_SHADER_SPV_DIR "/skybox.frag.spv");

    vk::ShaderModule vertModule = createShaderModule(device, vertCode);
    vk::ShaderModule fragModule = createShaderModule(device, fragCode);

    vk::PipelineShaderStageCreateInfo shaderStages[] = {
        { {}, vk::ShaderStageFlagBits::eVertex, vertModule, "main" },
        { {}, vk::ShaderStageFlagBits::eFragment, fragModule, "main" }
    };

    // No vertex input — the vertex shader generates a fullscreen triangle from gl_VertexIndex
    vk::PipelineVertexInputStateCreateInfo vertexInput({}, 0, nullptr, 0, nullptr);
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly(
        {}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
    vk::PipelineViewportStateCreateInfo viewportState({}, 1, nullptr, 1, nullptr);
    vk::PipelineRasterizationStateCreateInfo rasterizer(
        {}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill,
        vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise,
        VK_FALSE, 0, 0, 0, 1.0f);
    vk::PipelineMultisampleStateCreateInfo multisampling(
        {}, vk::SampleCountFlagBits::e1, VK_FALSE);

    // Depth test ON, depth write OFF — skybox is at z=1.0 (far plane),
    // particles and grid will overdraw it because they're closer.
    vk::PipelineDepthStencilStateCreateInfo depthStencil(
        {}, VK_TRUE, VK_FALSE, vk::CompareOp::eLessOrEqual, VK_FALSE, VK_FALSE);

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
        {}, 2, shaderStages, &vertexInput, &inputAssembly, nullptr,
        &viewportState, &rasterizer, &multisampling, &depthStencil,
        &colorBlending, &dynamicState, pipelineLayout, offscreenPass, 0);

    auto result = device.createGraphicsPipeline(nullptr, pipelineInfo);
    skyboxPipeline = result.value;

    device.destroyShaderModule(vertModule);
    device.destroyShaderModule(fragModule);
    Log::ok("Skybox pipeline created");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// IBL — IMAGE-BASED LIGHTING (split-sum precompute)
//
// The PBR shaders need three precomputed textures to light meshes from the
// environment map (see kmrb_lighting.glsl / kmrb_ibl):
//   irradiance  (32³ cube)        — cosine-convolved env  → diffuse ambient
//   prefiltered (128³ cube, mips) — GGX-convolved env     → specular ambient
//   BRDF LUT    (512² 2D)         — split-sum table, env-independent
//
// The images are fixed-size and live for the whole session; only their
// *contents* are re-baked (compute dispatch) when the HDR changes. The bake
// shaders write via storage-image views, which the main pipeline layout has
// no bindings for — hence the dedicated iblBake* layout/pipelines.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Renderer::createIBLResources(vk::Device device) {
    auto findMem = [&](vk::MemoryRequirements reqs) -> uint32_t {
        auto memProps = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((reqs.memoryTypeBits & (1 << i)) &&
                (memProps.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal))
                return i;
        }
        throw std::runtime_error("No device-local memory type for IBL images");
    };

    // RGBA16F everywhere: HDR range, and one of the few formats Vulkan
    // guarantees storage-image support for (compute shaders write these directly)
    const auto format = vk::Format::eR16G16B16A16Sfloat;
    const auto usage  = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage;

    auto makeImage = [&](uint32_t size, uint32_t mips, uint32_t layers, bool cube,
                         vk::Image& img, vk::DeviceMemory& mem) {
        vk::ImageCreateInfo info(
            cube ? vk::ImageCreateFlagBits::eCubeCompatible : vk::ImageCreateFlags{},
            vk::ImageType::e2D, format, vk::Extent3D(size, size, 1),
            mips, layers, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal,
            usage, vk::SharingMode::eExclusive);
        img = device.createImage(info);
        auto reqs = device.getImageMemoryRequirements(img);
        mem = device.allocateMemory(vk::MemoryAllocateInfo(reqs.size, findMem(reqs)));
        device.bindImageMemory(img, mem, 0);
    };

    // Samplers: clamp-to-edge, linear. The prefiltered sampler must allow all
    // mip levels — vk::SamplerCreateInfo defaults maxLod to 0, which would
    // silently clamp every textureLod() to the sharpest mip.
    auto makeSampler = [&](float maxLod) {
        vk::SamplerCreateInfo info({},
            vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear,
            vk::SamplerAddressMode::eClampToEdge, vk::SamplerAddressMode::eClampToEdge,
            vk::SamplerAddressMode::eClampToEdge);
        info.maxLod = maxLod;
        return device.createSampler(info);
    };

    // ── Irradiance cubemap (Set 1, binding 1) ──
    makeImage(IBL_IRRADIANCE_SIZE, 1, 6, true, iblIrradiance, iblIrradianceMemory);
    iblIrradianceView = device.createImageView(vk::ImageViewCreateInfo(
        {}, iblIrradiance, vk::ImageViewType::eCube, format, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)));
    // Compute shaders can't write through a cube view portably — they write the
    // same 6 layers through a 2D-array view instead (image2DArray in GLSL)
    iblIrradianceStorageView = device.createImageView(vk::ImageViewCreateInfo(
        {}, iblIrradiance, vk::ImageViewType::e2DArray, format, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 6)));
    iblIrradianceSampler = makeSampler(0.0f);

    // ── Prefiltered specular cubemap (Set 1, binding 2) ──
    makeImage(IBL_PREFILTERED_SIZE, IBL_PREFILTER_MIPS, 6, true,
              iblPrefiltered, iblPrefilteredMemory);
    iblPrefilteredView = device.createImageView(vk::ImageViewCreateInfo(
        {}, iblPrefiltered, vk::ImageViewType::eCube, format, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, IBL_PREFILTER_MIPS, 0, 6)));
    for (uint32_t mip = 0; mip < IBL_PREFILTER_MIPS; mip++) {
        // One write-target view per mip — each mip is a separate bake dispatch
        iblPrefilteredMipViews[mip] = device.createImageView(vk::ImageViewCreateInfo(
            {}, iblPrefiltered, vk::ImageViewType::e2DArray, format, {},
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, mip, 1, 0, 6)));
    }
    iblPrefilteredSampler = makeSampler(VK_LOD_CLAMP_NONE);

    // ── BRDF LUT (Set 1, binding 3) ──
    makeImage(IBL_BRDF_LUT_SIZE, 1, 1, false, iblBrdfLut, iblBrdfLutMemory);
    iblBrdfLutView = device.createImageView(vk::ImageViewCreateInfo(
        {}, iblBrdfLut, vk::ImageViewType::e2D, format, {},
        vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1)));
    iblBrdfLutSampler = makeSampler(0.0f);

    // ── Bake descriptor/pipeline layout: src env (sampled) + dst (storage) ──
    std::array<vk::DescriptorSetLayoutBinding, 2> bakeBindings = {{
        { 0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute },
        { 1, vk::DescriptorType::eStorageImage,         1, vk::ShaderStageFlagBits::eCompute }
    }};
    iblBakeSetLayout = device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo({},
            static_cast<uint32_t>(bakeBindings.size()), bakeBindings.data()));

    vk::PushConstantRange bakePush(vk::ShaderStageFlagBits::eCompute, 0, 16);
    iblBakePipelineLayout = device.createPipelineLayout(
        vk::PipelineLayoutCreateInfo({}, 1, &iblBakeSetLayout, 1, &bakePush));

    auto makeComputePipeline = [&](const char* spvName) {
        auto code = readFile(std::string(KMRB_SHADER_SPV_DIR "/") + spvName);
        vk::ShaderModule module = createShaderModule(device, code);
        vk::ComputePipelineCreateInfo info({},
            vk::PipelineShaderStageCreateInfo(
                {}, vk::ShaderStageFlagBits::eCompute, module, "main"),
            iblBakePipelineLayout);
        auto result = device.createComputePipeline(nullptr, info);
        device.destroyShaderModule(module);
        return result.value;
    };
    iblBrdfPipeline       = makeComputePipeline("brdf_lut.comp.spv");
    iblIrradiancePipeline = makeComputePipeline("irradiance.comp.spv");
    iblPrefilterPipeline  = makeComputePipeline("prefilter.comp.spv");

    // ── One-time BRDF LUT bake — depends only on the BRDF, never on the HDR ──
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, 1, &iblBakeSetLayout);
    vk::DescriptorSet lutSet = device.allocateDescriptorSets(allocInfo)[0];
    vk::DescriptorImageInfo lutDst({}, iblBrdfLutView, vk::ImageLayout::eGeneral);
    vk::WriteDescriptorSet lutWrite(lutSet, 1, 0, 1,
        vk::DescriptorType::eStorageImage, &lutDst);
    device.updateDescriptorSets(lutWrite, nullptr);

    vk::Queue queue = device.getQueue(graphicsQueueFamily, 0);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1))[0];
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // undefined → general (the only layout storage writes are allowed in)
    vk::ImageMemoryBarrier toGeneral(
        {}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        iblBrdfLut, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe,
        vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, toGeneral);

    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, iblBrdfPipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, iblBakePipelineLayout,
        0, lutSet, nullptr);
    cmd.dispatch(IBL_BRDF_LUT_SIZE / 8, IBL_BRDF_LUT_SIZE / 8, 1);  // local_size = 8x8

    // general → shader-read so fragment shaders can sample it
    vk::ImageMemoryBarrier toRead(
        vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        iblBrdfLut, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eFragmentShader, {}, nullptr, nullptr, toRead);

    cmd.end();
    queue.submit(vk::SubmitInfo({}, {}, cmd));
    queue.waitIdle();
    device.freeCommandBuffers(commandPool, cmd);
    device.freeDescriptorSets(descriptorPool, lutSet);

    writeIBLDescriptors(device);
    Log::ok("IBL resources created (BRDF LUT baked)");
}

// Point Set 1 bindings 1-3 at the IBL images. Called at init and again if the
// descriptor pool is rebuilt (envDescriptorSet gets reallocated then).
void Renderer::writeIBLDescriptors(vk::Device device) {
    vk::DescriptorImageInfo irrInfo(iblIrradianceSampler, iblIrradianceView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::DescriptorImageInfo preInfo(iblPrefilteredSampler, iblPrefilteredView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    vk::DescriptorImageInfo lutInfo(iblBrdfLutSampler, iblBrdfLutView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    std::array<vk::WriteDescriptorSet, 3> writes = {{
        { envDescriptorSet, 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &irrInfo },
        { envDescriptorSet, 2, 0, 1, vk::DescriptorType::eCombinedImageSampler, &preInfo },
        { envDescriptorSet, 3, 0, 1, vk::DescriptorType::eCombinedImageSampler, &lutInfo }
    }};
    device.updateDescriptorSets(writes, nullptr);
}

// Re-convolve irradiance + prefiltered from whatever envMap currently holds
// (real HDR or the black placeholder). Called after every load/clear — both
// call sites have already waitIdle'd, so the images are safe to overwrite.
void Renderer::bakeIBLMaps(vk::Device device) {
    // One descriptor set per dispatch: irradiance + one per prefilter mip.
    // They all share the same layout; sets can't be rewritten between
    // dispatches inside a single command buffer, hence one each.
    std::vector<vk::DescriptorSetLayout> layouts(1 + IBL_PREFILTER_MIPS, iblBakeSetLayout);
    auto bakeSets = device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(
        descriptorPool, static_cast<uint32_t>(layouts.size()), layouts.data()));

    vk::DescriptorImageInfo srcInfo(envSampler, envCubemapView,
        vk::ImageLayout::eShaderReadOnlyOptimal);
    std::vector<vk::DescriptorImageInfo> dstInfos;
    dstInfos.reserve(1 + IBL_PREFILTER_MIPS);
    dstInfos.emplace_back(vk::Sampler{}, iblIrradianceStorageView, vk::ImageLayout::eGeneral);
    for (uint32_t mip = 0; mip < IBL_PREFILTER_MIPS; mip++)
        dstInfos.emplace_back(vk::Sampler{}, iblPrefilteredMipViews[mip], vk::ImageLayout::eGeneral);

    std::vector<vk::WriteDescriptorSet> writes;
    for (size_t i = 0; i < bakeSets.size(); i++) {
        writes.push_back({ bakeSets[i], 0, 0, 1, vk::DescriptorType::eCombinedImageSampler, &srcInfo });
        writes.push_back({ bakeSets[i], 1, 0, 1, vk::DescriptorType::eStorageImage, &dstInfos[i] });
    }
    device.updateDescriptorSets(writes, nullptr);

    vk::Queue queue = device.getQueue(graphicsQueueFamily, 0);
    vk::CommandBuffer cmd = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1))[0];
    cmd.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

    // Both targets → general. oldLayout = undefined: we overwrite every texel,
    // so telling Vulkan to discard the old contents is free and always valid.
    auto barrier = [&](vk::Image img, uint32_t mips,
                       vk::AccessFlags srcAccess, vk::AccessFlags dstAccess,
                       vk::ImageLayout from, vk::ImageLayout to,
                       vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage) {
        vk::ImageMemoryBarrier b(srcAccess, dstAccess, from, to,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, img,
            vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, mips, 0, 6));
        cmd.pipelineBarrier(srcStage, dstStage, {}, nullptr, nullptr, b);
    };
    barrier(iblIrradiance, 1, {}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);
    barrier(iblPrefiltered, IBL_PREFILTER_MIPS, {}, vk::AccessFlagBits::eShaderWrite,
        vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral,
        vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eComputeShader);

    // Irradiance: one dispatch covers all 6 faces (z = layer)
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, iblIrradiancePipeline);
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, iblBakePipelineLayout,
        0, bakeSets[0], nullptr);
    cmd.dispatch(IBL_IRRADIANCE_SIZE / 8, IBL_IRRADIANCE_SIZE / 8, 6);

    // Prefilter: one dispatch per mip, roughness spread evenly over the chain
    cmd.bindPipeline(vk::PipelineBindPoint::eCompute, iblPrefilterPipeline);
    for (uint32_t mip = 0; mip < IBL_PREFILTER_MIPS; mip++) {
        float roughness = static_cast<float>(mip) / static_cast<float>(IBL_PREFILTER_MIPS - 1);
        cmd.pushConstants(iblBakePipelineLayout, vk::ShaderStageFlagBits::eCompute,
            0, sizeof(float), &roughness);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, iblBakePipelineLayout,
            0, bakeSets[1 + mip], nullptr);
        uint32_t mipSize = IBL_PREFILTERED_SIZE >> mip;  // 128, 64, 32, 16, 8 — all divisible by 8
        cmd.dispatch(mipSize / 8, mipSize / 8, 6);
    }

    // Both targets → shader-read for the PBR fragment shaders
    barrier(iblIrradiance, 1, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader);
    barrier(iblPrefiltered, IBL_PREFILTER_MIPS, vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
        vk::ImageLayout::eGeneral, vk::ImageLayout::eShaderReadOnlyOptimal,
        vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eFragmentShader);

    cmd.end();
    queue.submit(vk::SubmitInfo({}, {}, cmd));
    queue.waitIdle();
    device.freeCommandBuffers(commandPool, cmd);
    device.freeDescriptorSets(descriptorPool, bakeSets);

    Log::info("IBL maps baked (irradiance + prefiltered specular)");
}

void Renderer::destroyIBLResources(vk::Device device) {
    if (iblIrradianceView)        device.destroyImageView(iblIrradianceView);
    if (iblIrradianceStorageView) device.destroyImageView(iblIrradianceStorageView);
    if (iblIrradianceSampler)     device.destroySampler(iblIrradianceSampler);
    if (iblIrradiance)            device.destroyImage(iblIrradiance);
    if (iblIrradianceMemory)      device.freeMemory(iblIrradianceMemory);

    if (iblPrefilteredView)       device.destroyImageView(iblPrefilteredView);
    for (auto& view : iblPrefilteredMipViews)
        if (view) device.destroyImageView(view);
    if (iblPrefilteredSampler)    device.destroySampler(iblPrefilteredSampler);
    if (iblPrefiltered)           device.destroyImage(iblPrefiltered);
    if (iblPrefilteredMemory)     device.freeMemory(iblPrefilteredMemory);

    if (iblBrdfLutView)           device.destroyImageView(iblBrdfLutView);
    if (iblBrdfLutSampler)        device.destroySampler(iblBrdfLutSampler);
    if (iblBrdfLut)               device.destroyImage(iblBrdfLut);
    if (iblBrdfLutMemory)         device.freeMemory(iblBrdfLutMemory);

    if (iblBrdfPipeline)          device.destroyPipeline(iblBrdfPipeline);
    if (iblIrradiancePipeline)    device.destroyPipeline(iblIrradiancePipeline);
    if (iblPrefilterPipeline)     device.destroyPipeline(iblPrefilterPipeline);
    if (iblBakePipelineLayout)    device.destroyPipelineLayout(iblBakePipelineLayout);
    if (iblBakeSetLayout)         device.destroyDescriptorSetLayout(iblBakeSetLayout);
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
    for (auto& [key, inst] : meshInstances) destroyMeshInstance(device, inst);
    meshInstances.clear();
    meshCache.cleanup(bufferManager);
    if (envCubemapView) device.destroyImageView(envCubemapView);
    if (envSampler) device.destroySampler(envSampler);
    if (envCubemap) device.destroyImage(envCubemap);
    if (envCubemapMemory) device.freeMemory(envCubemapMemory);
    destroyIBLResources(device);
    if (skyboxPipeline) device.destroyPipeline(skyboxPipeline);
    device.destroyPipeline(gridPipeline);
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyRenderPass(offscreenPass);
    device.destroyRenderPass(swapchainPass);
}

} // namespace kmrb
