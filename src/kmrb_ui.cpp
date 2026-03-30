#include "kmrb_ui.hpp"
#include "kmrb_buffers.hpp"
#include "kmrb_sim.hpp"
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace kmrb {

// Convert hex color #RRGGBB to ImVec4 (0-1 range)
static ImVec4 hex(uint32_t rgb, float a = 1.0f) {
    return ImVec4(
        ((rgb >> 16) & 0xFF) / 255.0f,
        ((rgb >> 8)  & 0xFF) / 255.0f,
        ((rgb)       & 0xFF) / 255.0f,
        a
    );
}

static void applyKMRBTheme() {
    ImGuiStyle& s = ImGui::GetStyle();

    // ── Style settings ──
    s.WindowRounding    = 4.0f;
    s.ChildRounding     = 4.0f;
    s.FrameRounding     = 4.0f;
    s.PopupRounding     = 4.0f;
    s.ScrollbarRounding = 4.0f;
    s.GrabRounding      = 2.0f;
    s.TabRounding       = 4.0f;

    s.WindowPadding     = ImVec2(12, 12);
    s.FramePadding      = ImVec2(8, 4);
    s.ItemSpacing       = ImVec2(8, 6);
    s.ItemInnerSpacing  = ImVec2(6, 4);
    s.IndentSpacing     = 16.0f;

    s.ScrollbarSize     = 12.0f;
    s.GrabMinSize       = 8.0f;
    s.WindowBorderSize  = 1.0f;
    s.ChildBorderSize   = 1.0f;
    s.PopupBorderSize   = 1.0f;
    s.FrameBorderSize   = 0.0f;
    s.TabBorderSize     = 0.0f;

    s.WindowTitleAlign  = ImVec2(0.5f, 0.5f);

    // ── Backgrounds ──
    s.Colors[ImGuiCol_WindowBg]           = hex(0x0E0D0B, 0.94f);
    s.Colors[ImGuiCol_ChildBg]            = hex(0x1A1714);
    s.Colors[ImGuiCol_PopupBg]            = hex(0x1A1714);
    s.Colors[ImGuiCol_TitleBg]            = hex(0x252017);
    s.Colors[ImGuiCol_TitleBgActive]      = hex(0x30291E);
    s.Colors[ImGuiCol_TitleBgCollapsed]   = hex(0x1A1714);
    s.Colors[ImGuiCol_MenuBarBg]          = hex(0x252017);
    s.Colors[ImGuiCol_ScrollbarBg]        = hex(0x0E0D0B);

    // ── Tab ──
    s.Colors[ImGuiCol_Tab]                = hex(0x252017);
    s.Colors[ImGuiCol_TabSelected]        = hex(0x30291E);
    s.Colors[ImGuiCol_TabHovered]         = hex(0x3D352A);
    s.Colors[ImGuiCol_TabDimmed]          = hex(0x1A1714);
    s.Colors[ImGuiCol_TabDimmedSelected]  = hex(0x252017);
    s.Colors[ImGuiCol_TabSelectedOverline]= hex(0xC8A44E);

    // ── Headers ──
    s.Colors[ImGuiCol_Header]             = hex(0x252017);
    s.Colors[ImGuiCol_HeaderHovered]      = hex(0x30291E);
    s.Colors[ImGuiCol_HeaderActive]       = hex(0x3D352A);

    // ── Buttons ──
    s.Colors[ImGuiCol_Button]             = hex(0x252017);
    s.Colors[ImGuiCol_ButtonHovered]      = hex(0x30291E);
    s.Colors[ImGuiCol_ButtonActive]       = hex(0x3D352A);

    // ── Frame (input fields, slider bg) ──
    s.Colors[ImGuiCol_FrameBg]            = hex(0x1A1714);
    s.Colors[ImGuiCol_FrameBgHovered]     = hex(0x252017);
    s.Colors[ImGuiCol_FrameBgActive]      = hex(0x30291E);

    // ── Scrollbar ──
    s.Colors[ImGuiCol_ScrollbarGrab]         = hex(0x3D352A);
    s.Colors[ImGuiCol_ScrollbarGrabHovered]  = hex(0x5C5347);
    s.Colors[ImGuiCol_ScrollbarGrabActive]   = hex(0x8B7D6B);

    // ── God ray accents (golden) ──
    s.Colors[ImGuiCol_SliderGrab]         = hex(0xC8A44E);
    s.Colors[ImGuiCol_SliderGrabActive]   = hex(0xE2C36B);
    s.Colors[ImGuiCol_CheckMark]          = hex(0xC8A44E);
    s.Colors[ImGuiCol_PlotLines]          = hex(0xC8A44E);
    s.Colors[ImGuiCol_PlotHistogram]      = hex(0xC8A44E);
    s.Colors[ImGuiCol_TextSelectedBg]     = hex(0x2E2210);
    s.Colors[ImGuiCol_NavHighlight]       = hex(0xC8A44E);

    // ── Separators ──
    s.Colors[ImGuiCol_Separator]          = hex(0x3D352A);
    s.Colors[ImGuiCol_SeparatorHovered]   = hex(0x7A5F28);
    s.Colors[ImGuiCol_SeparatorActive]    = hex(0xC8A44E);

    // ── Resize grips ──
    s.Colors[ImGuiCol_ResizeGrip]         = hex(0x4A3818);
    s.Colors[ImGuiCol_ResizeGripHovered]  = hex(0x7A5F28);
    s.Colors[ImGuiCol_ResizeGripActive]   = hex(0xC8A44E);

    // ── Borders ──
    s.Colors[ImGuiCol_Border]             = hex(0x3D352A);
    s.Colors[ImGuiCol_BorderShadow]       = hex(0x000000, 0.0f);
    s.Colors[ImGuiCol_TableBorderStrong]  = hex(0x3D352A);
    s.Colors[ImGuiCol_TableBorderLight]   = hex(0x252017);
    s.Colors[ImGuiCol_TableHeaderBg]      = hex(0x252017);
    s.Colors[ImGuiCol_TableRowBg]         = hex(0x000000, 0.0f);
    s.Colors[ImGuiCol_TableRowBgAlt]      = hex(0x1A1714);

    // ── Text ──
    s.Colors[ImGuiCol_Text]               = hex(0xE8DCC8);
    s.Colors[ImGuiCol_TextDisabled]       = hex(0x5C5347);

    // ── Docking ──
    s.Colors[ImGuiCol_DockingPreview]     = hex(0xC8A44E, 0.4f);
    s.Colors[ImGuiCol_DockingEmptyBg]     = hex(0x0E0D0B);
}

void UI::init(GLFWwindow* window, vk::Instance instance, vk::PhysicalDevice physicalDevice,
              vk::Device device, uint32_t graphicsQueueFamily, vk::Queue graphicsQueue,
              vk::RenderPass renderPass, uint32_t imageCount) {

    // ImGui needs its own descriptor pool
    std::vector<vk::DescriptorPoolSize> poolSizes = {
        { vk::DescriptorType::eCombinedImageSampler, 100 }
    };

    vk::DescriptorPoolCreateInfo poolInfo(
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        100,
        static_cast<uint32_t>(poolSizes.size()), poolSizes.data()
    );

    imguiPool = device.createDescriptorPool(poolInfo);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Apply KMRB warm forest theme
    applyKMRBTheme();

    // GLFW backend
    ImGui_ImplGlfw_InitForVulkan(window, true);

    // Vulkan backend
    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device = device;
    initInfo.QueueFamily = graphicsQueueFamily;
    initInfo.Queue = graphicsQueue;
    initInfo.DescriptorPool = imguiPool;
    initInfo.MinImageCount = imageCount;
    initInfo.ImageCount = imageCount;
    initInfo.PipelineInfoMain.RenderPass = renderPass;
    initInfo.PipelineInfoMain.Subpass = 0;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;

    ImGui_ImplVulkan_Init(&initInfo);

    kmrb::Log::info("ImGui initialized");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// EDITOR LAYOUT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawEditorLayout(vk::DescriptorSet viewportTexture, vk::Extent2D viewportExtent,
                          uint32_t particleCount, float fps, float computeTime,
                          const std::unordered_map<std::string, BufferInfo>& buffers) {
    drawMenuBar();

    // Fullscreen dockspace — all panels dock into this
    ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());

    drawProjectBrowser();
    drawSceneHierarchy();
    drawViewport(viewportTexture, viewportExtent, particleCount, fps, computeTime);
    drawInspector(particleCount, buffers);
    drawConsole();
    drawDataOutput();

    if (showDemoWindow) ImGui::ShowDemoWindow(&showDemoWindow);
}

void UI::drawMenuBar() {
    if (ImGui::BeginMainMenuBar()) {

        // ── FILE ──
        if (ImGui::BeginMenu("File")) {

            if (ImGui::MenuItem("Open Scene...", "Ctrl+O")) {
                std::string path = openFileDialog(
                    "KMRB Scene (*.kmrb)\0*.kmrb\0All Files\0*.*\0", "Open Scene");
                if (!path.empty()) openScene(path);
            }

            if (ImGui::MenuItem("Save Scene", "Ctrl+S")) {
                if (currentScenePath.empty()) {
                    std::string path = saveFileDialog(
                        "KMRB Scene (*.kmrb)\0*.kmrb\0", "Save Scene");
                    if (!path.empty()) saveScene(path);
                } else {
                    saveScene(currentScenePath);
                }
            }

            if (ImGui::MenuItem("Save Scene As...")) {
                std::string path = saveFileDialog(
                    "KMRB Scene (*.kmrb)\0*.kmrb\0", "Save Scene As");
                if (!path.empty()) saveScene(path);
            }

            // Recent scenes submenu
            if (ImGui::BeginMenu("Recent Scenes", !recentScenes.empty())) {
                for (auto& scene : recentScenes) {
                    namespace fs = std::filesystem;
                    std::string name = fs::path(scene).filename().string();
                    if (ImGui::MenuItem(name.c_str())) {
                        openScene(scene);
                    }
                    if (ImGui::IsItemHovered()) {
                        ImGui::BeginTooltip();
                        ImGui::TextColored(hex(0x8B7D6B), "%s", scene.c_str());
                        ImGui::EndTooltip();
                    }
                }
                ImGui::Separator();
                if (ImGui::MenuItem("Clear Recent")) {
                    recentScenes.clear();
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("New Compute Shader")) {
                if (!projectRoot.empty()) {
                    namespace fs = std::filesystem;
                    std::string shadersDir = projectRoot + "/shaders/compute";
                    fs::create_directories(shadersDir);

                    std::string name = "custom.comp";
                    int n = 1;
                    while (fs::exists(shadersDir + "/" + name)) {
                        name = "custom_" + std::to_string(n++) + ".comp";
                    }

                    std::string path = shadersDir + "/" + name;
                    std::ofstream f(path);
                    f << "#version 450\n\nlayout(local_size_x = 256) in;\n\n"
                      << "layout(set = 0, binding = 0) uniform GlobalUBO {\n"
                      << "    mat4 view;\n    mat4 proj;\n    vec4 cameraPos;\n"
                      << "    float time;\n    float deltaTime;\n} global;\n\n"
                      << "struct Particle {\n    vec4 position;\n    vec4 velocity;\n    vec4 color;\n};\n\n"
                      << "layout(set = 2, binding = 0) readonly buffer ParticleInput {\n"
                      << "    Particle particles[];\n} ssboIn;\n\n"
                      << "layout(set = 2, binding = 1) writeonly buffer ParticleOutput {\n"
                      << "    Particle particles[];\n} ssboOut;\n\n"
                      << "void main() {\n"
                      << "    uint index = gl_GlobalInvocationID.x;\n"
                      << "    if (index >= ssboIn.particles.length()) return;\n\n"
                      << "    Particle p = ssboIn.particles[index];\n"
                      << "    float dt = global.deltaTime;\n\n"
                      << "    // Your simulation here\n\n"
                      << "    ssboOut.particles[index] = p;\n}\n";
                    f.close();
                    kmrb::Log::ok("Created shader: " + name);
                }
            }

            if (ImGui::MenuItem("Import Shader...")) {
                std::string path = openFileDialog(
                    "Compute Shader (*.comp)\0*.comp\0GLSL (*.glsl;*.vert;*.frag)\0*.glsl;*.vert;*.frag\0All Files\0*.*\0",
                    "Import Shader");
                if (!path.empty() && !projectRoot.empty()) {
                    namespace fs = std::filesystem;
                    std::string ext = fs::path(path).extension().string();
                    std::string destDir = projectRoot + "/shaders/"
                        + (ext == ".comp" ? "compute" : "render");
                    fs::create_directories(destDir);
                    std::string destPath = destDir + "/" + fs::path(path).filename().string();
                    fs::copy_file(path, destPath, fs::copy_options::overwrite_existing);
                    kmrb::Log::ok("Imported: " + fs::path(path).filename().string());
                }
            }

            ImGui::Separator();

            if (ImGui::BeginMenu("Export Data")) {
                if (ImGui::MenuItem("Particles to CSV...")) {
                    std::string path = saveFileDialog(
                        "CSV File (*.csv)\0*.csv\0", "Export Particles");
                    if (!path.empty() && onExportCSV) {
                        onExportCSV(path);
                    }
                }
                if (ImGui::MenuItem("Particles to EXR...")) {
                    kmrb::Log::info("EXR export not yet implemented");
                }
                ImGui::EndMenu();
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Quit", "Alt+F4")) {
                if (glfwWindow) glfwSetWindowShouldClose(glfwWindow, GLFW_TRUE);
            }

            ImGui::EndMenu();
        }

        // ── SIMULATION ──
        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem("Reset", "Ctrl+R")) {
                if (onReset) onReset();
            }
            ImGui::EndMenu();
        }

        // ── VIEW ──
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("ImGui Demo", nullptr, &showDemoWindow);
            ImGui::EndMenu();
        }

        ImGui::EndMainMenuBar();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PROJECT BROWSER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawProjectBrowser() {
    ImGui::SetNextWindowSize(ImVec2(200, 300), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Project")) {
        if (projectRoot.empty()) {
            ImGui::TextColored(hex(0x5C5347), "No project root set");
        } else {
            namespace fs = std::filesystem;

            // Show only content directories — shaders and scenes
            std::string shadersDir = projectRoot + "/shaders";
            std::string scenesDir = projectRoot + "/scenes";

            if (fs::exists(shadersDir) && ImGui::TreeNodeEx("shaders", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawFileTree(shadersDir);
                ImGui::TreePop();
            }

            if (fs::exists(scenesDir) && ImGui::TreeNodeEx("scenes", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawFileTree(scenesDir);
                ImGui::TreePop();
            } else {
                // Create scenes dir if it doesn't exist yet, show as empty
                if (ImGui::TreeNodeEx("scenes", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(hex(0x5C5347), "No scenes yet — .kmrb files go here");
                    ImGui::EndTooltip();
                }
            }
        }
    }
    ImGui::End();
}

void UI::drawFileTree(const std::string& directory) {
    namespace fs = std::filesystem;

    if (!fs::exists(directory) || !fs::is_directory(directory)) return;

    // Collect and sort entries (folders first, then files)
    std::vector<fs::directory_entry> folders, files;
    for (auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_directory()) {
            // Skip hidden/build/external dirs
            auto name = entry.path().filename().string();
            if (name[0] == '.' || name == "build" || name == "external") continue;
            folders.push_back(entry);
        } else {
            auto ext = entry.path().extension().string();
            // Only show shader and scene files
            if (ext == ".comp" || ext == ".vert" || ext == ".frag"
                || ext == ".glsl" || ext == ".kmrb" || ext == ".ksfx") {
                files.push_back(entry);
            }
        }
    }

    std::sort(folders.begin(), folders.end());
    std::sort(files.begin(), files.end());

    // Folders as collapsible tree nodes
    for (auto& folder : folders) {
        auto name = folder.path().filename().string();
        ImGuiTreeNodeFlags folderFlags = ImGuiTreeNodeFlags_OpenOnArrow;

        if (ImGui::TreeNodeEx(name.c_str(), folderFlags)) {
            drawFileTree(folder.path().string()); // Recurse
            ImGui::TreePop();
        }
    }

    // Files as selectable items
    for (auto& file : files) {
        auto name = file.path().filename().string();
        auto fullPath = file.path().string();
        auto ext = file.path().extension().string();

        bool isSelected = (selectedFile == fullPath);
        ImGuiTreeNodeFlags fileFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (isSelected) fileFlags |= ImGuiTreeNodeFlags_Selected;

        // Color code by file type
        if (ext == ".comp") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(0xC8A44E));  // Gold for compute
        } else if (ext == ".vert" || ext == ".frag") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(0x5A9BD4));  // Blue for render shaders
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(0xE8DCC8));  // Primary text
        }

        ImGui::TreeNodeEx(name.c_str(), fileFlags);

        // Drag-drop source — drag shader files onto Inspector slots
        if (ext == ".comp" || ext == ".vert" || ext == ".frag") {
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("KMRB_SHADER_PATH", fullPath.c_str(), fullPath.size() + 1);
                ImGui::TextColored(hex(0xC8A44E), "%s", name.c_str());
                ImGui::EndDragDropSource();
            }
        }

        // Single click = select
        if (ImGui::IsItemClicked()) {
            selectedFile = fullPath;
        }

        // Double click = open in system editor (VS Code, Notepad++, etc.)
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
            std::string cmd = "start \"\" \"" + fullPath + "\"";
            _popen(cmd.c_str(), "r");
            kmrb::Log::info("Opening in editor: " + name);
        }

        // Right-click context menu
        std::string popupId = "ctx_" + fullPath;
        if (ImGui::BeginPopupContextItem(popupId.c_str())) {
            if (ImGui::MenuItem("Open in Editor")) {
                std::string cmd = "start \"\" \"" + fullPath + "\"";
                _popen(cmd.c_str(), "r");
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Show in Explorer")) {
                // Windows explorer needs backslashes
                std::string winPath = fullPath;
                std::replace(winPath.begin(), winPath.end(), '/', '\\');
                std::string cmd = "explorer /select,\"" + winPath + "\"";
                system(cmd.c_str());
            }
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, hex(0xD46B5A)); // error-text red
            if (ImGui::MenuItem("Delete")) {
                std::filesystem::remove(fullPath);
                if (selectedFile == fullPath) selectedFile.clear();
                kmrb::Log::warn("Deleted: " + name);
            }
            ImGui::PopStyleColor();
            ImGui::EndPopup();
        }

        // Tooltip
        if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen(popupId.c_str())) {
            ImGui::BeginTooltip();
            ImGui::TextColored(hex(0x8B7D6B), "%s", fullPath.c_str());
            if (ext == ".comp" || ext == ".vert" || ext == ".frag") {
                ImGui::TextColored(hex(0x5C5347), "Drag onto Inspector to attach");
                ImGui::TextColored(hex(0x5C5347), "Double-click: open in editor");
            }
            ImGui::EndTooltip();
        }

        ImGui::PopStyleColor();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SCENE HIERARCHY
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawSceneHierarchy() {
    ImGui::SetNextWindowSize(ImVec2(200, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Scene Hierarchy")) {
        if (!registry) {
            ImGui::TextColored(hex(0x5C5347), "No scene loaded");
            ImGui::End();
            return;
        }

        // "+" button to add entities
        if (ImGui::SmallButton("+")) {
            ImGui::OpenPopup("add_entity_popup");
        }
        ImGui::SameLine();
        ImGui::TextColored(hex(0x5C5347), "Add Entity");

        drawAddEntityMenu();
        ImGui::Separator();

        // Helper to draw a single entity row
        entt::entity entityToDelete = entt::null;

        auto drawEntity = [&](entt::entity entity, const char* label, SelectionType type) {
            ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen
                                     | ImGuiTreeNodeFlags_SpanAvailWidth;
            if (selectedEntity == entity) flags |= ImGuiTreeNodeFlags_Selected;

            auto* name = registry->try_get<Name>(entity);
            std::string display = name ? name->value : label;

            // Active camera gets a gold marker
            if (type == SelectionType::Camera) {
                auto* cam = registry->try_get<CameraComponent>(entity);
                if (cam && cam->active) {
                    display += " *";
                }
            }

            ImGui::TreeNodeEx(reinterpret_cast<void*>(static_cast<uintptr_t>(static_cast<uint32_t>(entity))),
                              flags, "%s", display.c_str());

            if (ImGui::IsItemClicked()) {
                selectedEntity = entity;
                selectionType = type;
            }

            // Right-click context menu per entity
            if (ImGui::BeginPopupContextItem()) {
                if (type == SelectionType::Camera) {
                    auto* cam = registry->try_get<CameraComponent>(entity);
                    if (cam && !cam->active) {
                        if (ImGui::MenuItem("Set as Active Camera")) {
                            // Deactivate all other cameras
                            auto camView = registry->view<CameraComponent>();
                            for (auto e : camView) {
                                camView.get<CameraComponent>(e).active = false;
                            }
                            cam->active = true;
                            Log::info("Active camera switched to: " + display);
                        }
                    }
                }

                // Rename
                if (ImGui::MenuItem("Rename")) {
                    // TODO: inline rename popup
                }

                ImGui::Separator();
                ImGui::PushStyleColor(ImGuiCol_Text, hex(0xD46B5A));
                if (ImGui::MenuItem("Delete")) {
                    entityToDelete = entity;
                }
                ImGui::PopStyleColor();
                ImGui::EndPopup();
            }
        };

        // Scene root
        ImGuiTreeNodeFlags rootFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow;
        if (ImGui::TreeNodeEx("Scene", rootFlags)) {

            // Cameras
            auto cameraView = registry->view<CameraComponent, Name>();
            for (auto entity : cameraView) {
                drawEntity(entity, "Camera", SelectionType::Camera);
            }

            // Pipelines
            auto pipelineView = registry->view<PipelineComponent, Name>();
            for (auto entity : pipelineView) {
                drawEntity(entity, "Pipeline", SelectionType::Pipeline);
            }


            // Grids
            auto gridView = registry->view<GridComponent, Name>();
            for (auto entity : gridView) {
                drawEntity(entity, "Grid", SelectionType::Grid);
            }

            ImGui::TreePop();
        }

        // Deferred deletion (can't destroy during iteration)
        if (entityToDelete != entt::null) {
            auto* name = registry->try_get<Name>(entityToDelete);
            std::string dname = name ? name->value : "entity";
            if (selectedEntity == entityToDelete) {
                selectedEntity = entt::null;
                selectionType = SelectionType::None;
            }
            registry->destroy(entityToDelete);
            Log::warn("Deleted: " + dname);
        }
    }
    ImGui::End();
}

void UI::drawAddEntityMenu() {
    if (!registry) return;

    // Shared menu content for both popup triggers
    auto drawMenuItems = [&]() {
        if (ImGui::MenuItem("Pipeline")) {
            auto entity = registry->create();
            int count = 0;
            registry->view<PipelineComponent>().each([&](auto) { count++; });
            std::string name = count == 0 ? "Pipeline" : "Pipeline " + std::to_string(count + 1);

            registry->emplace<Name>(entity, name);
            registry->emplace<Transform>(entity);
            registry->emplace<PipelineComponent>(entity, PipelineComponent{10000});
            registry->emplace<ShaderProgramComponent>(entity);

            selectedEntity = entity;
            selectionType = SelectionType::Pipeline;
            Log::ok("Created: " + name);
        }

        if (ImGui::MenuItem("Camera")) {
            auto entity = registry->create();
            int count = 0;
            registry->view<CameraComponent>().each([&](auto) { count++; });
            std::string name = "Camera " + std::to_string(count + 1);

            registry->emplace<Name>(entity, name);
            registry->emplace<Transform>(entity, Transform{
                {0.0f, 2.0f, 5.0f}, {-15.0f, -90.0f, 0.0f}, {1.0f, 1.0f, 1.0f}});
            registry->emplace<CameraComponent>(entity);
            if (count == 0) registry->get<CameraComponent>(entity).active = true;

            selectedEntity = entity;
            selectionType = SelectionType::Camera;
            Log::ok("Created: " + name);
        }

        if (ImGui::MenuItem("Grid Helper")) {
            auto entity = registry->create();
            int count = 0;
            registry->view<GridComponent>().each([&](auto) { count++; });
            std::string name = count == 0 ? "Grid" : "Grid " + std::to_string(count + 1);

            registry->emplace<Name>(entity, name);
            registry->emplace<Transform>(entity);
            registry->emplace<GridComponent>(entity);

            selectedEntity = entity;
            selectionType = SelectionType::Grid;
            Log::ok("Created: " + name);
        }
    };

    // "+" button popup
    if (ImGui::BeginPopup("add_entity_popup")) {
        ImGui::TextColored(hex(0x8B7D6B), "Add Entity");
        ImGui::Separator();
        drawMenuItems();
        ImGui::EndPopup();
    }

    // Right-click empty space popup
    if (ImGui::BeginPopupContextWindow("hierarchy_context", ImGuiPopupFlags_NoOpenOverItems)) {
        ImGui::TextColored(hex(0x8B7D6B), "Add Entity");
        ImGui::Separator();
        drawMenuItems();
        ImGui::EndPopup();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VIEWPORT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawViewport(vk::DescriptorSet viewportTexture, vk::Extent2D viewportExtent,
                      uint32_t particleCount, float fps, float computeTime) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
    if (ImGui::Begin("Viewport")) {
        viewportHovered = ImGui::IsWindowHovered();

        // Display the offscreen render as an image filling the panel
        ImVec2 avail = ImGui::GetContentRegionAvail();
        if (viewportTexture && avail.x > 0 && avail.y > 0) {
            ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<VkDescriptorSet>(viewportTexture)), avail);
        }

        // Status bar overlay at the bottom of the viewport
        ImVec2 windowPos = ImGui::GetWindowPos();
        ImVec2 windowSize = ImGui::GetWindowSize();
        float barHeight = 24.0f;
        ImVec2 barPos(windowPos.x, windowPos.y + windowSize.y - barHeight);

        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(barPos, ImVec2(barPos.x + windowSize.x, barPos.y + barHeight),
            IM_COL32(14, 13, 11, 200)); // base bg

        ImGui::SetCursorScreenPos(ImVec2(barPos.x + 12, barPos.y + 4));

        // FPS
        ImGui::TextColored(hex(0x8B7D6B), "FPS");
        ImGui::SameLine();
        ImGui::TextColored(hex(0xE8DCC8), "%.0f", fps);
        ImGui::SameLine(0, 20);

        // Particle count
        ImGui::TextColored(hex(0x8B7D6B), "Particles");
        ImGui::SameLine();
        ImGui::TextColored(hex(0xE8DCC8), "%u", particleCount);
        ImGui::SameLine(0, 20);

        // Resolution
        ImGui::TextColored(hex(0x8B7D6B), "Res");
        ImGui::SameLine();
        ImGui::TextColored(hex(0xE8DCC8), "%ux%u", viewportExtent.width, viewportExtent.height);
    } else {
        viewportHovered = false;
    }
    ImGui::End();
    ImGui::PopStyleVar();
}

void UI::drawInspector(uint32_t particleCount,
                       const std::unordered_map<std::string, BufferInfo>& buffers) {
    ImGui::SetNextWindowSize(ImVec2(280, 500), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Inspector")) {
        if (!registry || selectedEntity == entt::null || !registry->valid(selectedEntity)) {
            ImGui::TextColored(hex(0x5C5347), "No entity selected");
            ImGui::End();
            return;
        }

        // ── Name (editable) ──
        auto* name = registry->try_get<Name>(selectedEntity);
        if (name) {
            char buf[128];
            strncpy(buf, name->value.c_str(), sizeof(buf));
            buf[sizeof(buf) - 1] = '\0';
            if (ImGui::InputText("##name", buf, sizeof(buf))) {
                name->value = buf;
            }
        }

        // ── Transform (universal) ──
        auto* transform = registry->try_get<Transform>(selectedEntity);
        if (transform) {
            if (ImGui::CollapsingHeader("Transform", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat3("Position", &transform->position.x, 0.1f);
                ImGui::DragFloat3("Rotation", &transform->rotation.x, 0.5f, -360.0f, 360.0f);
                ImGui::DragFloat3("Scale", &transform->scale.x, 0.01f, 0.01f, 100.0f);
            }
        }

        // ── Camera ──
        auto* cam = registry->try_get<CameraComponent>(selectedEntity);
        if (cam) {
            if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
                bool wasActive = cam->active;
                ImGui::Checkbox("Active", &cam->active);
                if (cam->active && !wasActive) {
                    auto camView = registry->view<CameraComponent>();
                    for (auto e : camView) {
                        if (e != selectedEntity)
                            camView.get<CameraComponent>(e).active = false;
                    }
                    Log::info("Active camera: " + (name ? name->value : "Camera"));
                }
                ImGui::SliderFloat("FOV", &cam->fov, 10.0f, 120.0f, "%.0f");
                ImGui::DragFloat("Near", &cam->nearPlane, 0.01f, 0.001f, 10.0f, "%.3f");
                ImGui::DragFloat("Far", &cam->farPlane, 1.0f, 1.0f, 10000.0f, "%.0f");
            }
        }

        // ── Pipeline ──
        auto* pl = registry->try_get<PipelineComponent>(selectedEntity);
        if (pl) {
            if (ImGui::CollapsingHeader("Pipeline", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::TextColored(hex(0x8B7D6B), "Particles");
                ImGui::SameLine(110);
                ImGui::TextColored(hex(0xE8DCC8), "%u", pl->particleCount);

                ImGui::TextColored(hex(0x8B7D6B), "Status");
                ImGui::SameLine(110);
                ImGui::TextColored(hex(0x7BA56E), "Running");

                ImGui::Spacing();

                // Shader program slots
                auto* shaderProg = registry->try_get<ShaderProgramComponent>(selectedEntity);
                if (shaderProg) {
                    ImGui::TextColored(hex(0xC8A44E), "Shader Program");

                    // Helper lambda for a shader slot with drag-drop target
                    auto drawShaderSlot = [&](const char* label, std::string& path, const char* ext) {
                        namespace fs = std::filesystem;
                        std::string filename = path.empty() ? "(none)" : fs::path(path).filename().string();
                        ImGui::TextColored(hex(0x8B7D6B), "  %s", label);
                        ImGui::SameLine(110);
                        ImGui::TextColored(path.empty() ? hex(0x5C5347) : hex(0xE8DCC8), "%s", filename.c_str());

                        // Clear button
                        if (!path.empty()) {
                            ImGui::SameLine();
                            std::string clearId = std::string("##clear_") + label;
                            if (ImGui::SmallButton(("x" + clearId).c_str())) {
                                path.clear();
                                shaderProg->dirty = true;
                            }
                        }

                        // Drag-drop target
                        if (ImGui::BeginDragDropTarget()) {
                            if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("KMRB_SHADER_PATH")) {
                                std::string droppedPath(static_cast<const char*>(payload->Data));
                                std::string droppedExt = fs::path(droppedPath).extension().string();
                                if (droppedExt == ext) {
                                    path = droppedPath;
                                    shaderProg->dirty = true;
                                    Log::ok("Attached: " + fs::path(droppedPath).filename().string());
                                } else {
                                    Log::warn("Wrong shader type for this slot");
                                }
                            }
                            ImGui::EndDragDropTarget();
                        }
                    };

                    drawShaderSlot("Compute", shaderProg->computePath, ".comp");
                    drawShaderSlot("Vertex", shaderProg->vertexPath, ".vert");
                    drawShaderSlot("Fragment", shaderProg->fragmentPath, ".frag");

                    if (ImGui::Button("Recompile")) {
                        shaderProg->dirty = true;
                    }
                    ImGui::Spacing();
                }


                if (gpuSupportsF64) {
                    if (ImGui::Checkbox("Float64 (double precision)", &f64Enabled)) {
                        Log::info(f64Enabled ? "Float64 ENABLED" : "Float64 DISABLED");
                    }
                } else {
                    ImGui::TextColored(hex(0xD46B5A), "Float64 not supported");
                }

                ImGui::Spacing();

                auto formatSize = [](vk::DeviceSize bytes) -> std::string {
                    if (bytes >= 1024 * 1024)
                        return std::to_string(bytes / (1024 * 1024)) + "." +
                               std::to_string((bytes % (1024 * 1024)) * 10 / (1024 * 1024)) + " MB";
                    if (bytes >= 1024)
                        return std::to_string(bytes / 1024) + "." +
                               std::to_string((bytes % 1024) * 10 / 1024) + " KB";
                    return std::to_string(bytes) + " B";
                };

                auto itA = buffers.find("particle_a");
                if (itA != buffers.end()) {
                    auto& pa = itA->second;
                    ImGui::TextColored(hex(0x8B7D6B), "SSBO size");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(0xE8DCC8), "%s (x2)", formatSize(pa.size).c_str());

                    ImGui::TextColored(hex(0x8B7D6B), "Stride");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(0xE8DCC8), "%u bytes", pa.elementStride);

                    ImGui::TextColored(hex(0x8B7D6B), "Precision");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(0xE8DCC8), "%s", f64Enabled ? "float64" : "float32");
                }

                ImGui::Spacing();
                if (ImGui::Button("Export to CSV")) {
                    std::string path = saveFileDialog("CSV File (*.csv)\0*.csv\0", "Export Particles");
                    if (!path.empty() && onExportCSV) onExportCSV(path);
                }
            }
        }

        // ── Grid ──
        auto* grid = registry->try_get<GridComponent>(selectedEntity);
        if (grid) {
            if (ImGui::CollapsingHeader("Grid", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::DragFloat("Size", &grid->size, 0.5f, 1.0f, 100.0f);
                ImGui::DragInt("Cells", &grid->cellCount, 1, 1, 100);
                ImGui::ColorEdit4("Color", &grid->color.x);
                ImGui::TextColored(hex(0x5C5347), "Cell size: %.2f", grid->size / grid->cellCount);
            }
        }


        // ── Light (stub for V2) ──
        auto* light = registry->try_get<LightComponent>(selectedEntity);
        if (light) {
            if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                const char* types[] = { "Point", "Directional", "Spot" };
                int lt = static_cast<int>(light->type);
                ImGui::Combo("Type", &lt, types, IM_ARRAYSIZE(types));
                light->type = static_cast<LightType>(lt);
                ImGui::ColorEdit3("Color", &light->color.x);
                ImGui::DragFloat("Intensity", &light->intensity, 0.1f, 0.0f, 100.0f);
                if (light->type != LightType::Directional) {
                    ImGui::DragFloat("Radius", &light->radius, 0.5f, 0.0f, 1000.0f);
                }
                if (light->type == LightType::Spot) {
                    ImGui::SliderFloat("Spot Angle", &light->spotAngle, 1.0f, 90.0f);
                }
                ImGui::TextColored(hex(0x5C5347), "Rendering not implemented (V2)");
            }
        }

        // ── Mesh Renderer (stub for V1 wireframe) ──
        auto* mesh = registry->try_get<MeshRendererComponent>(selectedEntity);
        if (mesh) {
            if (ImGui::CollapsingHeader("Mesh Renderer", ImGuiTreeNodeFlags_DefaultOpen)) {
                const char* shapes[] = { "Cube", "Sphere", "Plane" };
                int s = static_cast<int>(mesh->shape);
                ImGui::Combo("Shape", &s, shapes, IM_ARRAYSIZE(shapes));
                mesh->shape = static_cast<PrimitiveShape>(s);
                ImGui::ColorEdit4("Color", &mesh->color.x);
                ImGui::Checkbox("Wireframe", &mesh->wireframe);
                ImGui::TextColored(hex(0x5C5347), "Rendering not implemented yet");
            }
        }
    }
    ImGui::End();
}

void UI::drawConsole() {
    ImGui::SetNextWindowSize(ImVec2(500, 200), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Console")) {
        // Clear button
        if (ImGui::SmallButton("Clear")) Log::clear();
        ImGui::SameLine();
        ImGui::TextColored(hex(0x5C5347), "(%zu entries)", Log::getEntries().size());
        ImGui::Separator();

        // Scrollable log
        ImGui::BeginChild("log_scroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

        for (auto& entry : Log::getEntries()) {
            // Timestamp
            int mins = static_cast<int>(entry.timestamp) / 60;
            float secs = entry.timestamp - mins * 60;
            ImGui::TextColored(hex(0x5C5347), "%02d:%05.2f", mins, secs);
            ImGui::SameLine();

            // Level tag with color
            switch (entry.level) {
                case LogLevel::Info:
                    ImGui::TextColored(hex(0x5A9BD4), "[INF]"); break;
                case LogLevel::Ok:
                    ImGui::TextColored(hex(0x7BA56E), "[OK]");  break;
                case LogLevel::Warn:
                    ImGui::TextColored(hex(0xC8A44E), "[WRN]"); break;
                case LogLevel::Error:
                    ImGui::TextColored(hex(0xD46B5A), "[ERR]"); break;
            }
            ImGui::SameLine();

            // Message — errors in red, rest in primary text
            if (entry.level == LogLevel::Error)
                ImGui::TextColored(hex(0xD46B5A), "%s", entry.message.c_str());
            else
                ImGui::TextColored(hex(0xE8DCC8), "%s", entry.message.c_str());
        }

        // Auto-scroll to bottom when new entries arrive
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20)
            ImGui::SetScrollHereY(1.0f);

        ImGui::EndChild();
    }
    ImGui::End();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DATA OUTPUT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawDataOutput() {
    ImGui::SetNextWindowSize(ImVec2(600, 250), ImGuiCond_FirstUseEver);

    if (ImGui::Begin("Data Output")) {
        if (ImGui::BeginTabBar("data_tabs")) {

            // ── Buffer Table tab: live SSBO read-back ──
            if (ImGui::BeginTabItem("Buffer Table")) {

                // Refresh controls
                ImGui::Checkbox("Auto-refresh", &dataAutoRefresh);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(120);
                ImGui::SliderFloat("Interval (s)", &dataRefreshInterval, 0.1f, 2.0f, "%.1f");
                ImGui::SameLine();
                if (ImGui::SmallButton("Refresh Now")) {
                    dataRefreshTimer = dataRefreshInterval; // Force immediate refresh
                }

                // Timer-based read-back from GPU
                if (dataAutoRefresh && bufferManager) {
                    dataRefreshTimer += ImGui::GetIO().DeltaTime;
                    if (dataRefreshTimer >= dataRefreshInterval) {
                        dataRefreshTimer = 0.0f;
                        if (bufferManager->exists("particle_b")) {
                            cachedParticleData = bufferManager->readBack("particle_b");
                            cachedElementCount = bufferManager->getInfo("particle_b").elementCount;
                        }
                    }
                }

                ImGui::Separator();

                if (cachedParticleData.empty()) {
                    ImGui::TextColored(hex(0x5C5347), "No data — waiting for first refresh...");
                } else {
                    ImGui::TextColored(hex(0x5C5347), "%u particles", cachedElementCount);

                    // Scrollable particle data table
                    // Particle SSBO layout: 12 floats per particle
                    //   [0] pos.x  [1] pos.y  [2] pos.z  [3] pointSize
                    //   [4] vel.x  [5] vel.y  [6] vel.z  [7] lifetime
                    //   [8] r      [9] g      [10] b     [11] a
                    constexpr int FLOATS_PER_PARTICLE = 12;

                    ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollY
                                               | ImGuiTableFlags_RowBg
                                               | ImGuiTableFlags_BordersOuter
                                               | ImGuiTableFlags_BordersV
                                               | ImGuiTableFlags_Resizable
                                               | ImGuiTableFlags_Reorderable;

                    if (ImGui::BeginTable("particle_table", 8, tableFlags)) {
                        ImGui::TableSetupScrollFreeze(0, 1); // Freeze header row
                        ImGui::TableSetupColumn("ID",    ImGuiTableColumnFlags_WidthFixed, 50.0f);
                        ImGui::TableSetupColumn("pos.x", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("pos.y", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("pos.z", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.x", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.y", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.z", ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("mass",  ImGuiTableColumnFlags_WidthFixed, 60.0f);
                        ImGui::TableHeadersRow();

                        // Use clipper for 10k+ rows — only renders visible rows
                        ImGuiListClipper clipper;
                        clipper.Begin(static_cast<int>(cachedElementCount));
                        while (clipper.Step()) {
                            for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; row++) {
                                uint32_t offset = row * FLOATS_PER_PARTICLE;
                                if (offset + FLOATS_PER_PARTICLE > cachedParticleData.size()) break;

                                ImGui::TableNextRow();
                                ImGui::TableSetColumnIndex(0);
                                ImGui::Text("%d", row);
                                ImGui::TableSetColumnIndex(1);
                                ImGui::Text("%.3f", cachedParticleData[offset + 0]);
                                ImGui::TableSetColumnIndex(2);
                                ImGui::Text("%.3f", cachedParticleData[offset + 1]);
                                ImGui::TableSetColumnIndex(3);
                                ImGui::Text("%.3f", cachedParticleData[offset + 2]);
                                ImGui::TableSetColumnIndex(4);
                                ImGui::Text("%.3f", cachedParticleData[offset + 4]);
                                ImGui::TableSetColumnIndex(5);
                                ImGui::Text("%.3f", cachedParticleData[offset + 5]);
                                ImGui::TableSetColumnIndex(6);
                                ImGui::Text("%.3f", cachedParticleData[offset + 6]);
                                ImGui::TableSetColumnIndex(7);
                                ImGui::Text("%.2f", cachedParticleData[offset + 3]); // pointSize as mass
                            }
                        }
                        ImGui::EndTable();
                    }
                }
                ImGui::EndTabItem();
            }

            // ── Export tab ──
            if (ImGui::BeginTabItem("Export")) {
                ImGui::Spacing();

                // Export path with browse button
                ImGui::TextColored(hex(0x8B7D6B), "Export Path");
                char pathBuf[512];
                strncpy(pathBuf, exportPath.c_str(), sizeof(pathBuf));
                pathBuf[sizeof(pathBuf) - 1] = '\0';
                ImGui::SetNextItemWidth(-80);
                if (ImGui::InputText("##export_path", pathBuf, sizeof(pathBuf))) {
                    exportPath = pathBuf;
                }
                ImGui::SameLine();
                if (ImGui::Button("Browse")) {
                    std::string path = saveFileDialog("CSV File (*.csv)\0*.csv\0", "Export Particles");
                    if (!path.empty()) exportPath = path;
                }

                ImGui::Spacing();

                // Frame range (V1: exports current frame only, range stored for future use)
                ImGui::TextColored(hex(0x8B7D6B), "Frame Range");
                ImGui::SetNextItemWidth(100);
                ImGui::InputInt("Start", &exportFrameStart);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(100);
                ImGui::InputInt("End", &exportFrameEnd);
                if (exportFrameStart < 0) exportFrameStart = 0;
                if (exportFrameEnd < exportFrameStart) exportFrameEnd = exportFrameStart;

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                // CSV export button
                if (ImGui::Button("Export CSV", ImVec2(120, 0))) {
                    if (exportPath.empty()) {
                        // No path set — open dialog
                        std::string path = saveFileDialog("CSV File (*.csv)\0*.csv\0", "Export Particles");
                        if (!path.empty()) {
                            exportPath = path;
                            if (onExportCSV) onExportCSV(exportPath);
                        }
                    } else {
                        if (onExportCSV) onExportCSV(exportPath);
                    }
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// LOG SYSTEM
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

std::deque<LogEntry> Log::entries;
std::chrono::steady_clock::time_point Log::startTime = std::chrono::steady_clock::now();

void Log::add(LogLevel level, const std::string& msg) {
    float t = std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();
    entries.push_back({ t, level, msg });
    if (entries.size() > MAX_ENTRIES) entries.pop_front();
}

void Log::info(const std::string& msg)  { add(LogLevel::Info, msg); }
void Log::ok(const std::string& msg)    { add(LogLevel::Ok, msg); }
void Log::warn(const std::string& msg)  { add(LogLevel::Warn, msg); }
void Log::error(const std::string& msg) { add(LogLevel::Error, msg); }
void Log::clear()                       { entries.clear(); }
const std::deque<LogEntry>& Log::getEntries() { return entries; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FRAME LIFECYCLE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FILE DIALOGS & SCENE I/O
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>

std::string UI::openFileDialog(const char* filter, const char* title) {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwWindow ? glfwGetWin32Window(glfwWindow) : nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (GetOpenFileNameA(&ofn)) return std::string(filename);
    return "";
}

std::string UI::saveFileDialog(const char* filter, const char* title) {
    char filename[MAX_PATH] = "";
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = glfwWindow ? glfwGetWin32Window(glfwWindow) : nullptr;
    ofn.lpstrFilter = filter;
    ofn.lpstrFile = filename;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title;
    ofn.Flags = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

    if (GetSaveFileNameA(&ofn)) return std::string(filename);
    return "";
}

void UI::addRecentScene(const std::string& path) {
    // Remove if already in list
    auto it = std::find(recentScenes.begin(), recentScenes.end(), path);
    if (it != recentScenes.end()) recentScenes.erase(it);

    recentScenes.push_front(path);
    if (recentScenes.size() > MAX_RECENT_SCENES)
        recentScenes.pop_back();
}

void UI::openScene(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        kmrb::Log::error("Scene not found: " + path);
        return;
    }
    currentScenePath = path;
    addRecentScene(path);
    // TODO: parse .kmrb scene file and load entities/shaders/camera
    kmrb::Log::info("Opened scene: " + std::filesystem::path(path).filename().string());
}

void UI::saveScene(const std::string& path) {
    currentScenePath = path;
    addRecentScene(path);

    // Write a basic .kmrb scene file (JSON placeholder)
    std::ofstream f(path);
    f << "{\n";
    f << "  \"name\": \"" << std::filesystem::path(path).stem().string() << "\",\n";
    f << "  \"shader\": \"compute\",\n";
    f << "  \"particles\": 10000,\n";
    f << "  \"precision\": \"" << (f64Enabled ? "float64" : "float32") << "\"\n";
    f << "}\n";
    f.close();

    kmrb::Log::ok("Saved scene: " + std::filesystem::path(path).filename().string());
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FRAME LIFECYCLE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void UI::endFrame() {
    ImGui::Render();
}

void UI::render(vk::CommandBuffer cmd) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
}

void UI::onSwapchainRecreate(uint32_t newImageCount) {
    ImGui_ImplVulkan_SetMinImageCount(newImageCount);
}

void UI::cleanup(vk::Device device) {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    device.destroyDescriptorPool(imguiPool);
    kmrb::Log::info("ImGui shutdown");
}

} // namespace kmrb
