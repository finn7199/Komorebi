#include "kmrb_core.hpp"
#include <iostream>
#include <set>
#include <algorithm>

namespace kmrb {

void Core::init() {
    initWindow();
    createInstance();
    createSurface();       // Must be before pickPhysicalDevice (need surface for present checks)
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    renderer.init(device, physicalDevice, swapchainImageFormat, swapchainExtent,
                  swapchainImageViews, indices.graphicsFamily.value());
}

void Core::initWindow() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(1280, 720, "Komorebi", nullptr, nullptr);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window");
    }

    // Store `this` pointer so the static GLFW callback can reach our instance
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow* w, int, int) {
        auto* core = static_cast<Core*>(glfwGetWindowUserPointer(w));
        core->framebufferResized = true;
    });
}

void Core::createInstance() {
    if (enableValidationLayers && !checkValidationLayerSupport()) {
        throw std::runtime_error("KMRB: Validation layers requested but not available!");
    }

    vk::ApplicationInfo appInfo(
        "Komorebi",    VK_MAKE_VERSION(1, 0, 0),
        "KMRB_Engine", VK_MAKE_VERSION(1, 0, 0),
        VK_API_VERSION_1_3
    );

    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    vk::InstanceCreateInfo createInfo(
        {},
        &appInfo,
        enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0,
        enableValidationLayers ? validationLayers.data() : nullptr,
        static_cast<uint32_t>(extensions.size()),
        extensions.data()
    );

    instance = vk::createInstance(createInfo);
    std::cout << "[KMRB] Vulkan Instance created (API 1.3)" << std::endl;
}

bool Core::checkValidationLayerSupport() {
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

    for (const char* layerName : validationLayers) {
        bool found = false;
        for (const auto& layerProps : availableLayers) {
            if (strcmp(layerName, layerProps.layerName) == 0) {
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SURFACE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Surface = bridge between Vulkan and the OS window (platform-specific)
// GLFW handles the platform details (Win32/X11/Wayland) for us
void Core::createSurface() {
    // GLFW uses the C API (VkSurfaceKHR), so we pass the raw handles
    VkSurfaceKHR rawSurface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &rawSurface) != VK_SUCCESS) {
        throw std::runtime_error("KMRB: Failed to create window surface!");
    }
    surface = rawSurface;
    std::cout << "[KMRB] Window surface created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PHYSICAL DEVICE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Core::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();

    if (devices.empty()) {
        throw std::runtime_error("KMRB: No GPUs with Vulkan support found!");
    }

    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            physicalDevice = device;
            break;
        }
    }

    if (!physicalDevice) {
        throw std::runtime_error("KMRB: No suitable GPU found!");
    }

    vk::PhysicalDeviceProperties props = physicalDevice.getProperties();
    std::cout << "[KMRB] GPU selected: " << props.deviceName << std::endl;
}

bool Core::isDeviceSuitable(vk::PhysicalDevice device) {
    vk::PhysicalDeviceProperties props = device.getProperties();
    vk::PhysicalDeviceFeatures features = device.getFeatures();
    QueueFamilyIndices indices = findQueueFamilies(device);

    // Check that the GPU supports VK_KHR_swapchain
    bool extensionsSupported = checkDeviceExtensionSupport(device);

    // Only check swapchain details if the extension exists
    bool swapchainAdequate = false;
    if (extensionsSupported) {
        SwapchainSupportDetails support = querySwapchainSupport(device);
        swapchainAdequate = !support.formats.empty() && !support.presentModes.empty();
    }

    return props.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
        && features.geometryShader
        && indices.isComplete()
        && extensionsSupported
        && swapchainAdequate;
}

bool Core::checkDeviceExtensionSupport(vk::PhysicalDevice device) {
    std::vector<vk::ExtensionProperties> available = device.enumerateDeviceExtensionProperties();
    std::set<std::string> required(deviceExtensions.begin(), deviceExtensions.end());

    for (const auto& ext : available) {
        required.erase(ext.extensionName);
    }
    return required.empty();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// QUEUE FAMILIES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QueueFamilyIndices Core::findQueueFamilies(vk::PhysicalDevice device) {
    QueueFamilyIndices indices;

    std::vector<vk::QueueFamilyProperties> queueFamilies = device.getQueueFamilyProperties();

    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
            indices.graphicsFamily = i;
        }

        // Check if this family can present to our surface
        // Graphics and present are often the same family, but not guaranteed
        if (device.getSurfaceSupportKHR(i, surface)) {
            indices.presentFamily = i;
        }

        if (indices.isComplete()) break;
    }

    return indices;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LOGICAL DEVICE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Core::createLogicalDevice() {
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    // Create queues for each unique family (graphics + present may be same index)
    std::set<uint32_t> uniqueFamilies = {
        indices.graphicsFamily.value(),
        indices.presentFamily.value()
    };

    float queuePriority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    for (uint32_t family : uniqueFamilies) {
        queueCreateInfos.push_back(vk::DeviceQueueCreateInfo({}, family, 1, &queuePriority));
    }

    vk::PhysicalDeviceFeatures deviceFeatures{};

    vk::DeviceCreateInfo createInfo(
        {},
        static_cast<uint32_t>(queueCreateInfos.size()),
        queueCreateInfos.data(),
        enableValidationLayers ? static_cast<uint32_t>(validationLayers.size()) : 0,
        enableValidationLayers ? validationLayers.data() : nullptr,
        static_cast<uint32_t>(deviceExtensions.size()), // Now enabling VK_KHR_swapchain
        deviceExtensions.data(),
        &deviceFeatures
    );

    device = physicalDevice.createDevice(createInfo);
    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    presentQueue = device.getQueue(indices.presentFamily.value(), 0);

    std::cout << "[KMRB] Logical device created" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SWAPCHAIN
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// Query what our surface + GPU combo supports
SwapchainSupportDetails Core::querySwapchainSupport(vk::PhysicalDevice device) {
    SwapchainSupportDetails details;
    details.capabilities = device.getSurfaceCapabilitiesKHR(surface);
    details.formats = device.getSurfaceFormatsKHR(surface);
    details.presentModes = device.getSurfacePresentModesKHR(surface);
    return details;
}

// Prefer SRGB 8-bit color (standard for non-HDR rendering)
vk::SurfaceFormatKHR Core::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) {
    for (const auto& fmt : formats) {
        if (fmt.format == vk::Format::eB8G8R8A8Srgb
            && fmt.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return fmt;
        }
    }
    return formats[0]; // Fallback to whatever is available
}

// Present mode = how images are queued for display
//   eFifo    = vsync (guaranteed available, caps to monitor refresh rate)
//   eMailbox = triple-buffered vsync (low latency, uses more power)
vk::PresentModeKHR Core::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& modes) {
    for (const auto& mode : modes) {
        if (mode == vk::PresentModeKHR::eMailbox) return mode;
    }
    return vk::PresentModeKHR::eFifo; // Always supported, good default
}

// Swap extent = resolution of swapchain images (usually matches window size)
vk::Extent2D Core::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
    // If the extent is already set (not max uint32), the surface dictates the size
    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return capabilities.currentExtent;
    }

    // Otherwise query GLFW for actual framebuffer size (accounts for DPI scaling)
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    vk::Extent2D extent = {
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height)
    };

    // Clamp to GPU's supported range
    extent.width = std::clamp(extent.width,
        capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    extent.height = std::clamp(extent.height,
        capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    return extent;
}

void Core::createSwapchain() {
    SwapchainSupportDetails support = querySwapchainSupport(physicalDevice);

    vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
    vk::Extent2D extent = chooseSwapExtent(support.capabilities);

    // Request one more than minimum for triple buffering (but don't exceed max)
    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR createInfo(
        {},
        surface,
        imageCount,
        surfaceFormat.format,
        surfaceFormat.colorSpace,
        extent,
        1,                                        // imageArrayLayers (1 unless stereoscopic 3D)
        vk::ImageUsageFlagBits::eColorAttachment  // We render directly to these images
    );

    // If graphics and present are different queue families, images must be shared
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
    uint32_t familyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

    if (indices.graphicsFamily != indices.presentFamily) {
        // eConcurrent = both queues can access images without ownership transfers
        // Simpler but slightly less performant than eExclusive
        createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
        createInfo.queueFamilyIndexCount = 2;
        createInfo.pQueueFamilyIndices = familyIndices;
    } else {
        // eExclusive = one queue family owns the image at a time (faster, most common)
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    createInfo.preTransform = support.capabilities.currentTransform; // No rotation/flip
    createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque; // Ignore window alpha
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE; // Don't render pixels hidden behind other windows

    swapchain = device.createSwapchainKHR(createInfo);

    // Retrieve the swapchain images — Vulkan owns these, we just get handles
    swapchainImages = device.getSwapchainImagesKHR(swapchain);
    swapchainImageFormat = surfaceFormat.format;
    swapchainExtent = extent;

    std::cout << "[KMRB] Swapchain created (" << extent.width << "x" << extent.height
              << ", " << swapchainImages.size() << " images)" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// IMAGE VIEWS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ImageView = how you access an image in a pipeline (format, aspect, mip levels)
void Core::createImageViews() {
    swapchainImageViews.resize(swapchainImages.size());

    for (size_t i = 0; i < swapchainImages.size(); i++) {
        vk::ImageViewCreateInfo viewInfo(
            {},
            swapchainImages[i],
            vk::ImageViewType::e2D,       // Treat as 2D texture
            swapchainImageFormat,
            {},                            // Component swizzle (identity = RGBA as-is)
            vk::ImageSubresourceRange(
                vk::ImageAspectFlagBits::eColor, // We want the color data (not depth/stencil)
                0, 1,  // Base mip level, level count
                0, 1   // Base array layer, layer count
            )
        );

        swapchainImageViews[i] = device.createImageView(viewInfo);
    }

    std::cout << "[KMRB] Image views created (" << swapchainImageViews.size() << ")" << std::endl;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MAIN LOOP & CLEANUP
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void Core::run() {
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        bool needsRecreate = renderer.drawFrame(device, swapchain, graphicsQueue, presentQueue);
        if (needsRecreate || framebufferResized) {
            framebufferResized = false;
            recreateSwapchain();
        }
    }
    device.waitIdle();
}

// Swapchain recreation — needed when window is resized or swapchain becomes stale
void Core::recreateSwapchain() {
    // Handle minimization — wait until window has a real size again
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwGetFramebufferSize(window, &width, &height);
        glfwWaitEvents();
    }

    device.waitIdle();

    cleanupSwapchain();
    createSwapchain();
    createImageViews();

    renderer.onSwapchainRecreate(device, swapchainExtent, swapchainImageViews);
}

void Core::cleanupSwapchain() {
    for (auto& view : swapchainImageViews) {
        device.destroyImageView(view);
    }
    device.destroySwapchainKHR(swapchain);
}

// Destroy in reverse creation order
void Core::cleanup() {
    renderer.cleanup(device);
    cleanupSwapchain();
    device.destroy();
    instance.destroySurfaceKHR(surface);
    instance.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
}

} // namespace kmrb
