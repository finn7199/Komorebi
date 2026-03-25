#pragma once

#include <vulkan/vulkan.hpp> // C++ Vulkan bindings (vk:: namespace, RAII, exceptions)
#include <GLFW/glfw3.h>
#include <vector>
#include <optional>

#include "kmrb_renderer.hpp"

namespace kmrb {

// Queue family indices we need — graphics for rendering, present for displaying
struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;
    bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

// What the surface + GPU combo supports for swapchain creation
struct SwapchainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;   // Min/max images, extents, transforms
    std::vector<vk::SurfaceFormatKHR> formats; // Pixel formats + color spaces
    std::vector<vk::PresentModeKHR> presentModes; // Vsync modes (FIFO, mailbox, etc.)
};

class Core {
public:
    void init();
    void run();
    void cleanup();

private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;

    // ── Vulkan Core ──
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue; // May be same queue as graphics (usually is on desktop GPUs)

    // ── Surface & Swapchain ──
    vk::SurfaceKHR surface;                    // Connection between Vulkan and the OS window
    vk::SwapchainKHR swapchain;                // Rotating queue of images to render into
    std::vector<vk::Image> swapchainImages;    // The actual images owned by the swapchain
    std::vector<vk::ImageView> swapchainImageViews; // "Lenses" into images — how pipelines access them
    vk::Format swapchainImageFormat;
    vk::Extent2D swapchainExtent;

    // ── Renderer ──
    Renderer renderer;

    // Device extensions we require
    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME // Needed to present rendered images to the surface
    };

    // Validation layers (debug only)
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    // ── Init Helpers ──
    void initWindow();
    void createInstance();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapchain();
    void createImageViews();
    void recreateSwapchain();
    void cleanupSwapchain();

    // ── Query Helpers ──
    bool isDeviceSuitable(vk::PhysicalDevice device);
    bool checkValidationLayerSupport();
    bool checkDeviceExtensionSupport(vk::PhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice device);
    SwapchainSupportDetails querySwapchainSupport(vk::PhysicalDevice device);

    // ── Swapchain Config Helpers ──
    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats);
    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& modes);
    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities);
};

} // namespace kmrb
