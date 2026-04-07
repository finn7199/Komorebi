#pragma once

#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <string>
#include <functional>
#include <unordered_map>
#include <vector>
#include <deque>
#include <chrono>
#include <mutex>
#include <entt/entt.hpp>

namespace kmrb { class BufferManager; }

namespace kmrb {

struct BufferInfo; // Forward declare

// What kind of entity is selected in the hierarchy
enum class SelectionType { None, Scene, Pipeline, Camera, Grid };

// Log levels matching the KMRB color palette
enum class LogLevel { Info, Ok, Warn, Error };

struct LogEntry {
    float timestamp;    // Seconds since app start
    LogLevel level;
    std::string message;
};

// Global log — call from anywhere, displayed in the Console panel
class Log {
public:
    static void info(const std::string& msg);
    static void ok(const std::string& msg);
    static void warn(const std::string& msg);
    static void error(const std::string& msg);
    static void clear();

    static const std::deque<LogEntry>& getEntries();

private:
    static void add(LogLevel level, const std::string& msg);
    static std::deque<LogEntry> entries;
    static std::chrono::steady_clock::time_point startTime;
    static constexpr size_t MAX_ENTRIES = 500;
};


class UI {
public:
    void init(GLFWwindow* window, vk::Instance instance, vk::PhysicalDevice physicalDevice,
              vk::Device device, uint32_t graphicsQueueFamily, vk::Queue graphicsQueue,
              vk::RenderPass renderPass, uint32_t imageCount);
    void cleanup(vk::Device device);

    void beginFrame();
    void endFrame();
    void render(vk::CommandBuffer cmd);

    // The KMRB editor layout — all panels
    void drawEditorLayout(vk::DescriptorSet viewportTexture, vk::Extent2D viewportExtent,
                          uint32_t particleCount, float fps, float computeTime,
                          const std::unordered_map<std::string, BufferInfo>& buffers);

    // True if the viewport panel is hovered (for camera input)
    bool isViewportHovered() const { return viewportHovered; }

    void setProjectRoot(const std::string& root) { projectRoot = root; }
    void setOnReset(std::function<void()> cb) { onReset = std::move(cb); }
    void setOnExportCSV(std::function<void(const std::string&)> cb) { onExportCSV = std::move(cb); }
    void setOnEnvMapLoad(std::function<void(const std::string&)> cb) { onEnvMapLoad = std::move(cb); }
    void setOnEnvMapClear(std::function<void()> cb) { onEnvMapClear = std::move(cb); }
    void setOnReloadShaders(std::function<void()> cb) { onReloadShaders = std::move(cb); }

    // Simulation state — read by renderer to gate compute dispatch
    bool shouldDispatchCompute() {
        if (simRunning) return true;
        if (stepRequested) { stepRequested = false; return true; }
        return false;
    }
    void setWindow(GLFWwindow* w) { glfwWindow = w; }
    void setRegistry(entt::registry* reg) { registry = reg; }
    void setBufferManager(BufferManager* bm) { bufferManager = bm; }

    // Shader instances map — gives the Inspector access to reflected push constant
    // params so it can auto-generate UI widgets. Set by Core after renderer init.
    // Stored as void* to avoid circular header dependency (Renderer → UI → Renderer).
    // Cast to std::unordered_map<uint32_t, Renderer::ShaderInstance>* in kmrb_ui.cpp.
    void setShaderInstances(void* si) { shaderInstancesPtr = si; }

    // Inspector state — read by renderer/core

    SelectionType getSelectionType() const { return selectionType; }
    float getCameraMoveSpeed() const { return cameraMoveSpeed; }
    float getCameraLookSensitivity() const { return cameraLookSensitivity; }
    int getRenderWidth() const { return renderWidth; }
    int getRenderHeight() const { return renderHeight; }
    int getParticleCount() const { return particleCount; }
    bool isRenderResolutionDirty() { bool d = renderResDirty; renderResDirty = false; return d; }
    bool isParticleCountDirty() { bool d = particleCountDirty; particleCountDirty = false; return d; }

    // Handle swapchain recreation
    void onSwapchainRecreate(uint32_t newImageCount);

private:
    vk::DescriptorPool imguiPool;

    bool showDemoWindow = false;
    bool showPreferences = false;
    bool viewportHovered = false;
    std::string projectRoot;
    std::string selectedFile;
    std::function<void()> onReset;
    std::function<void(const std::string&)> onExportCSV;
    std::function<void(const std::string&)> onEnvMapLoad;
    std::function<void()> onEnvMapClear;
    std::function<void()> onReloadShaders;
    GLFWwindow* glfwWindow = nullptr;

    // Simulation playback state
    bool simRunning = true;
    bool stepRequested = false;

    // Scene state
    std::string currentScenePath;
    std::deque<std::string> recentScenes;               // Last 10 opened scenes
    static constexpr size_t MAX_RECENT_SCENES = 10;

    // Inspector state


    // Shader reflection — opaque pointer to Renderer's shaderInstances map
    void* shaderInstancesPtr = nullptr;

    // Preferences
    int renderWidth = 1920;
    int renderHeight = 1080;
    bool renderResDirty = false;
    float cameraMoveSpeed = 5.0f;
    float cameraLookSensitivity = 0.15f;
    int particleCount = 10000;
    bool particleCountDirty = false;

    // Environment map — scene-level property
    std::string envMapPath;

    // Data Output panel state
    BufferManager* bufferManager = nullptr;
    std::vector<float> cachedParticleData;   // Last SSBO read-back
    float dataRefreshInterval = 0.5f;        // Seconds between refreshes
    float dataRefreshTimer = 0.0f;
    uint32_t cachedElementCount = 0;
    bool dataAutoRefresh = true;
    std::string exportPath;
    int exportFrameStart = 0;
    int exportFrameEnd = 0;

    // Scene hierarchy state
    entt::registry* registry = nullptr;
    entt::entity selectedEntity = entt::null;
    SelectionType selectionType = SelectionType::None;

    void drawMenuBar();
    void drawSceneHierarchy();
    void drawAddEntityMenu(); // "+" button / right-click context menu
    void drawEntityInspector(); // Context-sensitive entity properties
    void drawViewport(vk::DescriptorSet viewportTexture, vk::Extent2D viewportExtent,
                      uint32_t particleCount, float fps, float computeTime);
    void drawProjectBrowser();
    void drawFileTree(const std::string& directory);
    void drawInspector(uint32_t particleCount,
                       const std::unordered_map<std::string, BufferInfo>& buffers);
    void drawConsole();
    void drawDataOutput();
    void drawPreferences();

    // File dialog helpers (Windows native)
    std::string openFileDialog(const char* filter, const char* title);
    std::string saveFileDialog(const char* filter, const char* title);
    void addRecentScene(const std::string& path);
    void saveScene(const std::string& path);
    void openScene(const std::string& path);
};

} // namespace kmrb
