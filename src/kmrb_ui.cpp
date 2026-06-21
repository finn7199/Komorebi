#include "kmrb_ui.hpp"
#include "kmrb_buffers.hpp"
#include "kmrb_palette.hpp"
#include "kmrb_sim.hpp"
#include "kmrb_renderer.hpp"  // For ShaderInstance / ReflectedParam types in Inspector
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>   // Pulls in <windows.h>
#include <shellapi.h>           // ShellExecuteA
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>

namespace kmrb {

// Open a file with its associated application (VS Code, image viewer, ...).
// ShellExecuteA is async and handle-free — unlike _popen("start ...") which
// leaks a FILE*, or system() which blocks the render thread on a shell.
static void openInSystemEditor(const std::string& path) {
    ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}

// Human-readable label for a mesh entity's geometry source — shown in the Inspector
// and the Data Output asset list. Primitives display by name; files by filename.
static std::string meshDisplayLabel(const std::string& meshPath) {
    if (meshPath.empty())             return "(no mesh)";
    if (meshPath == PRIMITIVE_CUBE)   return "Cube";
    if (meshPath == PRIMITIVE_SPHERE) return "Sphere";
    return std::filesystem::path(meshPath).filename().string();
}

// Reveal a file in Windows Explorer (selects it in its folder)
static void showInExplorer(const std::string& path) {
    std::string winPath = path;
    std::replace(winPath.begin(), winPath.end(), '/', '\\');  // Explorer wants backslashes
    std::string args = "/select,\"" + winPath + "\"";
    ShellExecuteA(nullptr, "open", "explorer.exe", args.c_str(), nullptr, SW_SHOWNORMAL);
}

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
    s.Colors[ImGuiCol_WindowBg]           = hex(palette::Base, 0.94f);
    s.Colors[ImGuiCol_ChildBg]            = hex(palette::Panel);
    s.Colors[ImGuiCol_PopupBg]            = hex(palette::Panel);
    s.Colors[ImGuiCol_TitleBg]            = hex(palette::Raised);
    s.Colors[ImGuiCol_TitleBgActive]      = hex(palette::Hover);
    s.Colors[ImGuiCol_TitleBgCollapsed]   = hex(palette::Panel);
    s.Colors[ImGuiCol_MenuBarBg]          = hex(palette::Raised);
    s.Colors[ImGuiCol_ScrollbarBg]        = hex(palette::Base);

    // ── Tab ──
    s.Colors[ImGuiCol_Tab]                = hex(palette::Raised);
    s.Colors[ImGuiCol_TabSelected]        = hex(palette::Hover);
    s.Colors[ImGuiCol_TabHovered]         = hex(palette::Border);
    s.Colors[ImGuiCol_TabDimmed]          = hex(palette::Panel);
    s.Colors[ImGuiCol_TabDimmedSelected]  = hex(palette::Raised);
    s.Colors[ImGuiCol_TabSelectedOverline]= hex(palette::Gold);

    // ── Headers ──
    s.Colors[ImGuiCol_Header]             = hex(palette::Raised);
    s.Colors[ImGuiCol_HeaderHovered]      = hex(palette::Hover);
    s.Colors[ImGuiCol_HeaderActive]       = hex(palette::Border);

    // ── Buttons ──
    s.Colors[ImGuiCol_Button]             = hex(palette::Raised);
    s.Colors[ImGuiCol_ButtonHovered]      = hex(palette::Hover);
    s.Colors[ImGuiCol_ButtonActive]       = hex(palette::Border);

    // ── Frame (input fields, slider bg) ──
    s.Colors[ImGuiCol_FrameBg]            = hex(palette::Panel);
    s.Colors[ImGuiCol_FrameBgHovered]     = hex(palette::Raised);
    s.Colors[ImGuiCol_FrameBgActive]      = hex(palette::Hover);

    // ── Scrollbar ──
    s.Colors[ImGuiCol_ScrollbarGrab]         = hex(palette::Border);
    s.Colors[ImGuiCol_ScrollbarGrabHovered]  = hex(palette::TextDim);
    s.Colors[ImGuiCol_ScrollbarGrabActive]   = hex(palette::TextMuted);

    // ── God ray accents (golden) ──
    s.Colors[ImGuiCol_SliderGrab]         = hex(palette::Gold);
    s.Colors[ImGuiCol_SliderGrabActive]   = hex(palette::GoldBright);
    s.Colors[ImGuiCol_CheckMark]          = hex(palette::Gold);
    s.Colors[ImGuiCol_PlotLines]          = hex(palette::Gold);
    s.Colors[ImGuiCol_PlotHistogram]      = hex(palette::Gold);
    s.Colors[ImGuiCol_TextSelectedBg]     = hex(palette::GoldSelection);
    s.Colors[ImGuiCol_NavHighlight]       = hex(palette::Gold);

    // ── Separators ──
    s.Colors[ImGuiCol_Separator]          = hex(palette::Border);
    s.Colors[ImGuiCol_SeparatorHovered]   = hex(palette::GoldDim);
    s.Colors[ImGuiCol_SeparatorActive]    = hex(palette::Gold);

    // ── Resize grips ──
    s.Colors[ImGuiCol_ResizeGrip]         = hex(palette::GoldFaint);
    s.Colors[ImGuiCol_ResizeGripHovered]  = hex(palette::GoldDim);
    s.Colors[ImGuiCol_ResizeGripActive]   = hex(palette::Gold);

    // ── Borders ──
    s.Colors[ImGuiCol_Border]             = hex(palette::Border);
    s.Colors[ImGuiCol_BorderShadow]       = hex(palette::Black, 0.0f);
    s.Colors[ImGuiCol_TableBorderStrong]  = hex(palette::Border);
    s.Colors[ImGuiCol_TableBorderLight]   = hex(palette::Raised);
    s.Colors[ImGuiCol_TableHeaderBg]      = hex(palette::Raised);
    s.Colors[ImGuiCol_TableRowBg]         = hex(palette::Black, 0.0f);
    s.Colors[ImGuiCol_TableRowBgAlt]      = hex(palette::Panel);

    // ── Text ──
    s.Colors[ImGuiCol_Text]               = hex(palette::Text);
    s.Colors[ImGuiCol_TextDisabled]       = hex(palette::TextDim);

    // ── Docking ──
    s.Colors[ImGuiCol_DockingPreview]     = hex(palette::Gold, 0.4f);
    s.Colors[ImGuiCol_DockingEmptyBg]     = hex(palette::Base);
}

// Create a new shader in the project by copying an engine template from
// shaders/templates/ — plain GLSL files users can read and edit, instead of
// string literals buried in C++. Picks a unique name: base.ext, base_1.ext, ...
static void createShaderFromTemplate(const std::string& projectRoot,
                                     const char* templateName,
                                     const char* destSubdir,
                                     const char* baseName) {
    namespace fs = std::filesystem;
    if (projectRoot.empty()) return;

    fs::path templatePath = fs::path(KMRB_SHADER_DIR) / "templates" / templateName;
    if (!fs::exists(templatePath)) {
        kmrb::Log::error("Shader template missing: " + templatePath.generic_string());
        return;
    }

    std::string destDir = projectRoot + destSubdir;
    fs::create_directories(destDir);

    std::string ext = fs::path(templateName).extension().string();
    std::string name = std::string(baseName) + ext;
    int n = 1;
    while (fs::exists(destDir + "/" + name)) {
        name = std::string(baseName) + "_" + std::to_string(n++) + ext;
    }

    fs::copy_file(templatePath, destDir + "/" + name);
    kmrb::Log::ok("Created shader: " + name);
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

    if (showPreferences) drawPreferences();
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
                        ImGui::TextColored(hex(palette::TextMuted), "%s", scene.c_str());
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

            // "New X Shader" — copies an engine template (shaders/templates/)
            // into the project; templates are editable GLSL files on disk
            if (ImGui::MenuItem("New Init Shader")) {
                createShaderFromTemplate(projectRoot, "init.comp", "/shaders/compute", "init");
            }

            if (ImGui::MenuItem("New Compute Shader")) {
                createShaderFromTemplate(projectRoot, "compute.comp", "/shaders/compute", "custom");
            }

            if (ImGui::MenuItem("New Vertex Shader")) {
                createShaderFromTemplate(projectRoot, "particle.vert", "/shaders/render", "custom");
            }

            if (ImGui::MenuItem("New Fragment Shader")) {
                createShaderFromTemplate(projectRoot, "particle.frag", "/shaders/render", "custom");
            }

            if (ImGui::MenuItem("Import File...")) {
                std::string path = openFileDialog(
                    "All Supported\0*.comp;*.vert;*.frag;*.glsl;*.hdr;*.fbx;*.obj;*.gltf;*.glb;*.kmrb\0"
                    "Shaders (*.comp;*.vert;*.frag)\0*.comp;*.vert;*.frag\0"
                    "3D Models (*.fbx;*.obj;*.gltf;*.glb)\0*.fbx;*.obj;*.gltf;*.glb\0"
                    "HDR Images (*.hdr)\0*.hdr\0"
                    "All Files\0*.*\0",
                    "Import File");
                if (!path.empty() && !projectRoot.empty()) {
                    namespace fs = std::filesystem;
                    std::string ext = fs::path(path).extension().string();
                    std::string filename = fs::path(path).filename().string();

                    // Route files to their subdirectories
                    std::string destDir;
                    if (ext == ".comp") destDir = projectRoot + "/shaders/compute";
                    else if (ext == ".vert" || ext == ".frag") destDir = projectRoot + "/shaders/render";
                    else if (ext == ".fbx" || ext == ".obj" || ext == ".gltf" || ext == ".glb")
                        destDir = projectRoot + "/models";
                    else destDir = projectRoot;

                    fs::create_directories(destDir);
                    std::string destPath = destDir + "/" + filename;
                    fs::copy_file(path, destPath, fs::copy_options::overwrite_existing);
                    kmrb::Log::ok("Imported: " + filename);
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

        // ── EDIT ──
        if (ImGui::BeginMenu("Edit")) {
            if (ImGui::MenuItem("Preferences")) {
                showPreferences = true;
            }
            ImGui::EndMenu();
        }

        // ── SIMULATION ──
        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem(simRunning ? "Pause" : "Play", "Space")) {
                simRunning = !simRunning;
                Log::info(simRunning ? "Simulation playing" : "Simulation paused");
            }
            if (ImGui::MenuItem("Step Forward", "N", false, !simRunning)) {
                stepRequested = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Restart", "Ctrl+R")) {
                if (onReset) onReset();
            }
            if (ImGui::MenuItem("Reload Shaders")) {
                if (onReloadShaders) onReloadShaders();
            }
            ImGui::EndMenu();
        }

        // ── RENDER ──
        if (ImGui::BeginMenu("Render")) {
            if (ImGui::MenuItem("Reload Mesh Shaders")) {
                if (registry) {
                    int count = 0;
                    auto view = registry->view<MeshRendererComponent>();
                    for (auto e : view) {
                        view.get<MeshRendererComponent>(e).shaderDirty = true;
                        count++;
                    }
                    if (count > 0) Log::info("Reloading mesh shaders (" + std::to_string(count) + " mesh(es))...");
                    else           Log::warn("No mesh entities in scene");
                }
            }

            if (ImGui::MenuItem("Toggle Wireframe (All)")) {
                if (registry) {
                    auto view = registry->view<MeshRendererComponent>();
                    // If any mesh is solid, switch all to wireframe; otherwise switch all off
                    bool anyOff = false;
                    for (auto e : view) anyOff |= !view.get<MeshRendererComponent>(e).wireframe;
                    int count = 0;
                    for (auto e : view) {
                        auto& m = view.get<MeshRendererComponent>(e);
                        m.wireframe = anyOff;
                        m.shaderDirty = true;
                        count++;
                    }
                    if (count > 0) Log::info(std::string("Wireframe ") + (anyOff ? "on" : "off") + " (all meshes)");
                    else           Log::warn("No mesh entities in scene");
                }
            }

            ImGui::Separator();

            if (ImGui::MenuItem("Clear Mesh Cache")) {
                if (onClearMeshCache) onClearMeshCache();
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
            ImGui::TextColored(hex(palette::TextDim), "No project root set");
        } else {
            // Refresh the cached tree on a timer instead of scanning every frame
            projectTreeTimer += ImGui::GetIO().DeltaTime;
            if (projectTreeTimer >= PROJECT_TREE_REFRESH_SEC) {
                projectTreeTimer = 0.0f;
                refreshProjectTree();
            }

            if (!shaderTree.fullPath.empty() && ImGui::TreeNodeEx("shaders", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawFileTreeNodes(shaderTree.children);
                ImGui::TreePop();
            }

            if (!modelTree.fullPath.empty() && ImGui::TreeNodeEx("models", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawFileTreeNodes(modelTree.children);
                ImGui::TreePop();
            }

            // Root-level asset files (HDR images, etc.) — shown without a subfolder
            for (const auto& node : rootAssets) {
                const std::string& name = node.name;
                const std::string& fullPath = node.fullPath;

                ImGuiTreeNodeFlags fileFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
                if (selectedFile == fullPath) fileFlags |= ImGuiTreeNodeFlags_Selected;

                ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Cyan));  // Cyan for HDR
                ImGui::TreeNodeEx(name.c_str(), fileFlags);
                ImGui::PopStyleColor();

                // Drag-drop source for HDR files
                if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                    ImGui::SetDragDropPayload("KMRB_HDR_PATH", fullPath.c_str(), fullPath.size() + 1);
                    ImGui::TextColored(hex(palette::Cyan), "%s", name.c_str());
                    ImGui::EndDragDropSource();
                }

                if (ImGui::IsItemClicked()) selectedFile = fullPath;

                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(hex(palette::TextMuted), "%s", fullPath.c_str());
                    ImGui::TextColored(hex(palette::TextDim), "Drag onto Scene > Environment to set as skybox");
                    ImGui::EndTooltip();
                }
            }

            if (!sceneTree.fullPath.empty() && ImGui::TreeNodeEx("scenes", ImGuiTreeNodeFlags_DefaultOpen)) {
                drawFileTreeNodes(sceneTree.children);
                ImGui::TreePop();
            } else {
                // Create scenes dir if it doesn't exist yet, show as empty
                if (ImGui::TreeNodeEx("scenes", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen)) {
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::TextColored(hex(palette::TextDim), "No scenes yet — .kmrb files go here");
                    ImGui::EndTooltip();
                }
            }
        }
    }
    ImGui::End();
}

// Rebuild the cached file tree from disk. Called on a timer, not per frame.
void UI::refreshProjectTree() {
    namespace fs = std::filesystem;

    auto scanDir = [](const std::string& dir, FileTreeNode& node) {
        node = {};
        if (!fs::exists(dir) || !fs::is_directory(dir)) return;
        node.fullPath = dir;   // Non-empty fullPath = directory exists
        node.isDir = true;
        buildFileTree(dir, node);
    };

    scanDir(projectRoot + "/shaders", shaderTree);
    scanDir(projectRoot + "/models", modelTree);
    scanDir(projectRoot + "/scenes", sceneTree);

    // Root-level asset files (HDR images, etc.)
    rootAssets.clear();
    for (auto& entry : fs::directory_iterator(projectRoot)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension().string() != ".hdr") continue;
        FileTreeNode file;
        file.name = entry.path().filename().string();
        file.fullPath = entry.path().generic_string();  // Forward slashes
        file.ext = ".hdr";
        rootAssets.push_back(std::move(file));
    }
}

// Recursively snapshot a directory into FileTreeNode children (folders first, sorted)
void UI::buildFileTree(const std::string& directory, FileTreeNode& node) {
    namespace fs = std::filesystem;

    std::vector<FileTreeNode> folders, files;
    for (auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_directory()) {
            // Skip hidden/build/external dirs, engine shaders, and templates
            // (templates are copied via File > New Shader, not edited in place)
            auto name = entry.path().filename().string();
            if (name.empty() || name[0] == '.' || name == "build" || name == "external"
                || name == "engine" || name == "templates") continue;
            FileTreeNode folder;
            folder.name = name;
            folder.fullPath = entry.path().generic_string();
            folder.isDir = true;
            buildFileTree(folder.fullPath, folder); // Recurse
            folders.push_back(std::move(folder));
        } else {
            auto ext = entry.path().extension().string();
            // Only show shader, scene, HDR, and mesh files
            if (ext == ".comp" || ext == ".vert" || ext == ".frag"
                || ext == ".glsl" || ext == ".kmrb" || ext == ".ksfx" || ext == ".hdr"
                || ext == ".fbx" || ext == ".obj" || ext == ".gltf" || ext == ".glb") {
                FileTreeNode file;
                file.name = entry.path().filename().string();
                file.fullPath = entry.path().generic_string();
                file.ext = ext;
                files.push_back(std::move(file));
            }
        }
    }

    auto byName = [](const FileTreeNode& a, const FileTreeNode& b) { return a.name < b.name; };
    std::sort(folders.begin(), folders.end(), byName);
    std::sort(files.begin(), files.end(), byName);

    node.children.reserve(folders.size() + files.size());
    for (auto& f : folders) node.children.push_back(std::move(f));
    for (auto& f : files)   node.children.push_back(std::move(f));
}

// Draw cached tree nodes — no filesystem access here
void UI::drawFileTreeNodes(const std::vector<FileTreeNode>& nodes) {
    // Folders as collapsible tree nodes
    for (const auto& node : nodes) {
        if (!node.isDir) continue;
        if (ImGui::TreeNodeEx(node.name.c_str(), ImGuiTreeNodeFlags_OpenOnArrow)) {
            drawFileTreeNodes(node.children);
            ImGui::TreePop();
        }
    }

    // Files as selectable items
    for (const auto& node : nodes) {
        if (node.isDir) continue;
        const std::string& name = node.name;
        const std::string& fullPath = node.fullPath;
        const std::string& ext = node.ext;

        bool isSelected = (selectedFile == fullPath);
        ImGuiTreeNodeFlags fileFlags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen;
        if (isSelected) fileFlags |= ImGuiTreeNodeFlags_Selected;

        // Color code by file type
        if (ext == ".comp") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Gold));  // Gold for compute
        } else if (ext == ".vert" || ext == ".frag") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Blue));  // Blue for render shaders
        } else if (ext == ".hdr") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Cyan));  // Cyan for HDR env maps
        } else if (ext == ".fbx" || ext == ".obj" || ext == ".gltf" || ext == ".glb") {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Tan));  // Warm tan for 3D models
        } else {
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Text));  // Primary text
        }

        ImGui::TreeNodeEx(name.c_str(), fileFlags);

        // Drag-drop source — drag shader files onto Inspector slots
        if (ext == ".comp" || ext == ".vert" || ext == ".frag") {
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("KMRB_SHADER_PATH", fullPath.c_str(), fullPath.size() + 1);
                ImGui::TextColored(hex(palette::Gold), "%s", name.c_str());
                ImGui::EndDragDropSource();
            }
        }

        // Drag-drop source for HDR files — drag onto Scene > Environment slot
        if (ext == ".hdr") {
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("KMRB_HDR_PATH", fullPath.c_str(), fullPath.size() + 1);
                ImGui::TextColored(hex(palette::Cyan), "%s", name.c_str());
                ImGui::EndDragDropSource();
            }
        }

        // Drag-drop source for mesh files — drag onto Mesh entity Inspector slot
        if (ext == ".fbx" || ext == ".obj" || ext == ".gltf" || ext == ".glb") {
            if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
                ImGui::SetDragDropPayload("KMRB_MESH_PATH", fullPath.c_str(), fullPath.size() + 1);
                ImGui::TextColored(hex(palette::Tan), "%s", name.c_str());
                ImGui::EndDragDropSource();
            }
        }

        // Single click = select
        if (ImGui::IsItemClicked()) {
            selectedFile = fullPath;
        }

        // Double click = open in system editor (VS Code, Notepad++, etc.)
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
            openInSystemEditor(fullPath);
            kmrb::Log::info("Opening in editor: " + name);
        }

        // Right-click context menu
        std::string popupId = "ctx_" + fullPath;
        if (ImGui::BeginPopupContextItem(popupId.c_str())) {
            if (ImGui::MenuItem("Open in Editor")) {
                openInSystemEditor(fullPath);
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Show in Explorer")) {
                showInExplorer(fullPath);
            }
            ImGui::Separator();
            ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Red)); // error-text red
            if (ImGui::MenuItem("Delete")) {
                std::filesystem::remove(fullPath);
                if (selectedFile == fullPath) selectedFile.clear();
                projectTreeTimer = PROJECT_TREE_REFRESH_SEC;  // Refresh tree next frame
                kmrb::Log::warn("Deleted: " + name);
            }
            ImGui::PopStyleColor();
            ImGui::EndPopup();
        }

        // Tooltip
        if (ImGui::IsItemHovered() && !ImGui::IsPopupOpen(popupId.c_str())) {
            ImGui::BeginTooltip();
            ImGui::TextColored(hex(palette::TextMuted), "%s", fullPath.c_str());
            if (ext == ".comp" || ext == ".vert" || ext == ".frag") {
                ImGui::TextColored(hex(palette::TextDim), "Drag onto Inspector to attach");
                ImGui::TextColored(hex(palette::TextDim), "Double-click: open in editor");
            } else if (ext == ".hdr") {
                ImGui::TextColored(hex(palette::TextDim), "Drag onto Scene > Environment to set as skybox");
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
            ImGui::TextColored(hex(palette::TextDim), "No scene loaded");
            ImGui::End();
            return;
        }

        // "+" button to add entities
        if (ImGui::SmallButton("+")) {
            ImGui::OpenPopup("add_entity_popup");
        }
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::TextDim), "Add Entity");

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
                ImGui::PushStyleColor(ImGuiCol_Text, hex(palette::Red));
                if (ImGui::MenuItem("Delete")) {
                    entityToDelete = entity;
                }
                ImGui::PopStyleColor();
                ImGui::EndPopup();
            }
        };

        // Scene root — clicking it shows scene-level settings in Inspector
        ImGuiTreeNodeFlags rootFlags = ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_OpenOnArrow
                                     | ImGuiTreeNodeFlags_SpanAvailWidth;
        if (selectionType == SelectionType::Scene) rootFlags |= ImGuiTreeNodeFlags_Selected;
        if (ImGui::TreeNodeEx("Scene", rootFlags)) {
            // Click on "Scene" label to select scene-level settings
            if (ImGui::IsItemClicked()) {
                selectedEntity = entt::null;
                selectionType = SelectionType::Scene;
            }

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

            // Meshes
            auto meshView = registry->view<MeshRendererComponent, Name>();
            for (auto entity : meshView) {
                drawEntity(entity, "Mesh", SelectionType::Mesh);
            }

            // Lights
            auto lightView = registry->view<LightComponent, Name>();
            for (auto entity : lightView) {
                drawEntity(entity, "Light", SelectionType::Light);
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
                CAMERA_SPAWN_POSITION, CAMERA_SPAWN_ROTATION, {1.0f, 1.0f, 1.0f}});
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

        if (ImGui::BeginMenu("Mesh")) {
            // Create a mesh entity. `meshPath` is empty for "Empty", or a "__primitive_*"
            // sentinel for a primitive (the renderer resolves it to the cached GPU mesh).
            // meshCacheKey stays empty so syncMeshInstances resolves it next frame.
            auto createMesh = [&](const char* base, const char* meshPath) {
                auto entity = registry->create();
                // Number per primitive type (same meshPath), not by total mesh count, so the
                // first Sphere is "Sphere" even if a Cube already exists.
                int count = 0;
                registry->view<MeshRendererComponent>().each([&](auto, auto& m) {
                    if (m.meshPath == meshPath) count++;
                });
                std::string name = count == 0 ? base : std::string(base) + " " + std::to_string(count + 1);

                registry->emplace<Name>(entity, name);
                registry->emplace<Transform>(entity);
                registry->emplace<MeshRendererComponent>(entity).meshPath = meshPath;

                selectedEntity = entity;
                selectionType = SelectionType::Mesh;
                Log::ok("Created: " + name);
            };

            if (ImGui::MenuItem("Empty"))  createMesh("Mesh",   "");
            if (ImGui::MenuItem("Cube"))   createMesh("Cube",   PRIMITIVE_CUBE);
            if (ImGui::MenuItem("Sphere")) createMesh("Sphere", PRIMITIVE_SPHERE);
            ImGui::EndMenu();
        }

        if (ImGui::MenuItem("Light")) {
            auto entity = registry->create();
            int count = 0;
            registry->view<LightComponent>().each([&](auto) { count++; });
            std::string name = count == 0 ? "Light" : "Light " + std::to_string(count + 1);

            registry->emplace<Name>(entity, name);
            registry->emplace<Transform>(entity, Transform{{2.0f, 3.0f, 2.0f}, {}, {1.0f, 1.0f, 1.0f}});
            registry->emplace<LightComponent>(entity);

            selectedEntity = entity;
            selectionType = SelectionType::Light;
            Log::ok("Created: " + name);
        }
    };

    // "+" button popup
    if (ImGui::BeginPopup("add_entity_popup")) {
        ImGui::TextColored(hex(palette::TextMuted), "Add Entity");
        ImGui::Separator();
        drawMenuItems();
        ImGui::EndPopup();
    }

    // Right-click empty space popup
    if (ImGui::BeginPopupContextWindow("hierarchy_context", ImGuiPopupFlags_NoOpenOverItems)) {
        ImGui::TextColored(hex(palette::TextMuted), "Add Entity");
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
    // Mesh stats for status bar — O(n meshes), trivially cheap
    uint32_t meshCount = 0, meshTotalVerts = 0;
    if (registry) {
        auto mv = registry->view<MeshRendererComponent>();
        for (auto e : mv) {
            meshCount++;
            meshTotalVerts += mv.get<MeshRendererComponent>(e).vertexCount;
        }
    }

    // Compact vertex count formatter: 48288 → "48.3K", 1200000 → "1.2M"
    auto fmtVerts = [](uint32_t v) -> std::string {
        char buf[16];
        if (v >= 1000000) { snprintf(buf, sizeof(buf), "%.1fM", v / 1000000.0f); return buf; }
        if (v >= 1000)    { snprintf(buf, sizeof(buf), "%.1fK", v / 1000.0f);    return buf; }
        return std::to_string(v);
    };

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
            ImGui::ColorConvertFloat4ToU32(hex(palette::Base, 200.0f / 255.0f)));

        ImGui::SetCursorScreenPos(ImVec2(barPos.x + 12, barPos.y + 4));

        // FPS
        ImGui::TextColored(hex(palette::TextMuted), "FPS");
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::Text), "%.0f", fps);
        ImGui::SameLine(0, 20);

        // Particle count
        ImGui::TextColored(hex(palette::TextMuted), "Particles");
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::Text), "%u", particleCount);
        ImGui::SameLine(0, 20);

        // Mesh count + aggregate vertex total
        ImGui::TextColored(hex(palette::TextMuted), "Meshes");
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::Text), "%u", meshCount);
        if (meshTotalVerts > 0) {
            ImGui::SameLine(0, 4);
            ImGui::TextColored(hex(palette::TextDim), "(%sv)", fmtVerts(meshTotalVerts).c_str());
        }
        ImGui::SameLine(0, 20);

        // Resolution: viewport panel size | render framebuffer size
        ImGui::TextColored(hex(palette::TextMuted), "Viewport");
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::Text), "%.0fx%.0f", avail.x, avail.y);
        ImGui::SameLine(0, 10);
        ImGui::TextColored(hex(palette::TextMuted), "Render");
        ImGui::SameLine();
        ImGui::TextColored(hex(palette::Text), "%dx%d", renderWidth, renderHeight);
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
        // ── Scene-level settings (no entity selected, Scene root clicked) ──
        if (selectionType == SelectionType::Scene) {
            ImGui::TextColored(hex(palette::Gold), "Scene Settings");
            ImGui::Separator();

            if (ImGui::CollapsingHeader("Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
                namespace fs = std::filesystem;
                std::string filename = envMapPath.empty() ? "(none)" : fs::path(envMapPath).filename().string();

                ImGui::TextColored(hex(palette::TextMuted), "HDR Map");
                ImGui::SameLine(110);
                ImGui::TextColored(envMapPath.empty() ? hex(palette::TextDim) : hex(palette::Text), "%s", filename.c_str());

                // Clear button
                if (!envMapPath.empty()) {
                    ImGui::SameLine();
                    if (ImGui::SmallButton("x##clear_env")) {
                        envMapPath.clear();
                        if (onEnvMapClear) onEnvMapClear();
                    }
                }

                // Drag-drop target for .hdr files
                if (ImGui::BeginDragDropTarget()) {
                    if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("KMRB_HDR_PATH")) {
                        std::string path(static_cast<const char*>(payload->Data));
                        envMapPath = path;
                        if (onEnvMapLoad) onEnvMapLoad(path);
                    }
                    ImGui::EndDragDropTarget();
                }

                if (envMapPath.empty()) {
                    ImGui::TextColored(hex(palette::TextDim), "Drag an .hdr file here from the Project Browser");
                }
            }

            ImGui::End();
            return;
        }

        if (!registry || selectedEntity == entt::null || !registry->valid(selectedEntity)) {
            ImGui::TextColored(hex(palette::TextDim), "No entity selected");
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
                ImGui::TextColored(hex(palette::TextMuted), "Particles");
                ImGui::SameLine(110);
                // Single source of truth for the particle count — the renderer
                // watches this component and reallocates the SSBOs on change.
                // Commit only when editing finishes (Enter / lose focus), not per
                // keystroke, since every change triggers a GPU buffer reallocation.
                // InputInt wraps InputScalar, which rejects EnterReturnsTrue — ImGui
                // directs us to IsItemDeactivatedAfterEdit() for this exact case.
                int count = static_cast<int>(pl->particleCount);
                ImGui::SetNextItemWidth(-1);
                ImGui::InputInt("##particle_count", &count, 1000, 10000);
                if (ImGui::IsItemDeactivatedAfterEdit()) {
                    pl->particleCount = static_cast<uint32_t>(std::clamp(count, 100, 1000000));
                }

                ImGui::TextColored(hex(palette::TextMuted), "Status");
                ImGui::SameLine(110);
                bool compiled = false;
                if (shaderInstancesPtr) {
                    auto* insts = static_cast<std::unordered_map<uint32_t, Renderer::ShaderInstance>*>(shaderInstancesPtr);
                    auto stIt = insts->find(static_cast<uint32_t>(selectedEntity));
                    compiled = stIt != insts->end() &&
                               (stIt->second.computePipeline || stIt->second.graphicsPipeline);
                }
                if (!compiled)       ImGui::TextColored(hex(palette::TextDim), "Not compiled");
                else if (simRunning) ImGui::TextColored(hex(palette::Green), "Running");
                else                 ImGui::TextColored(hex(palette::Gold), "Paused");

                ImGui::Spacing();

                // Shader program slots
                auto* shaderProg = registry->try_get<ShaderProgramComponent>(selectedEntity);
                if (shaderProg) {
                    ImGui::TextColored(hex(palette::Gold), "Shader Program");

                    // Helper lambda for a shader slot with drag-drop target
                    auto drawShaderSlot = [&](const char* label, std::string& path, const char* ext) {
                        namespace fs = std::filesystem;
                        std::string filename = path.empty() ? "(none)" : fs::path(path).filename().string();
                        ImGui::TextColored(hex(palette::TextMuted), "  %s", label);
                        ImGui::SameLine(110);
                        ImGui::TextColored(path.empty() ? hex(palette::TextDim) : hex(palette::Text), "%s", filename.c_str());

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

                    drawShaderSlot("Init", shaderProg->initPath, ".comp");
                    drawShaderSlot("Compute", shaderProg->computePath, ".comp");
                    drawShaderSlot("Vertex", shaderProg->vertexPath, ".vert");
                    drawShaderSlot("Fragment", shaderProg->fragmentPath, ".frag");

                    if (ImGui::Button("Recompile")) {
                        shaderProg->dirty = true;
                    }
                    ImGui::Spacing();
                }

                // ── Reflected Parameters ──
                // Auto-generated from shader push constants via SPIRV-Reflect.
                // Each user-defined param (offset >= 80) gets a matching ImGui widget.
                // The user can drag sliders or edit values, which writes directly into
                // the ShaderInstance's pushConstantData buffer — picked up by the GPU next frame.
                if (shaderInstancesPtr) {
                    auto* instances = static_cast<std::unordered_map<uint32_t, Renderer::ShaderInstance>*>(shaderInstancesPtr);
                    uint32_t entityKey = static_cast<uint32_t>(selectedEntity);
                    auto instIt = instances->find(entityKey);
                    if (instIt != instances->end()) {
                        auto& inst = instIt->second;
                        if (!inst.reflectedParams.empty()) {
                            ImGui::Spacing();
                            ImGui::TextColored(hex(palette::Gold), "Parameters");

                            for (size_t pi = 0; pi < inst.reflectedParams.size(); pi++) {
                                auto& param = inst.reflectedParams[pi];
                                // Get a pointer into the live push constant data at this param's offset
                                uint8_t* raw = inst.pushConstantData.data() + param.offset;

                                // Each widget needs a unique ImGui ID — use offset to guarantee uniqueness
                                ImGui::PushID(static_cast<int>(param.offset));

                                switch (param.type) {
                                    case Renderer::ReflectedParam::Float:
                                        ImGui::DragFloat(param.name.c_str(), reinterpret_cast<float*>(raw), 0.01f);
                                        break;
                                    case Renderer::ReflectedParam::Vec2:
                                        ImGui::DragFloat2(param.name.c_str(), reinterpret_cast<float*>(raw), 0.01f);
                                        break;
                                    case Renderer::ReflectedParam::Vec3:
                                        ImGui::DragFloat3(param.name.c_str(), reinterpret_cast<float*>(raw), 0.01f);
                                        break;
                                    case Renderer::ReflectedParam::Vec4:
                                        ImGui::DragFloat4(param.name.c_str(), reinterpret_cast<float*>(raw), 0.01f);
                                        break;
                                    case Renderer::ReflectedParam::Int:
                                        ImGui::DragInt(param.name.c_str(), reinterpret_cast<int*>(raw));
                                        break;
                                    case Renderer::ReflectedParam::Bool:
                                        ImGui::Checkbox(param.name.c_str(), reinterpret_cast<bool*>(raw));
                                        break;
                                    case Renderer::ReflectedParam::Mat4:
                                        // Matrices are not easily editable — just show the name
                                        ImGui::TextColored(hex(palette::TextDim), "%s (mat4)", param.name.c_str());
                                        break;
                                    default:
                                        ImGui::TextColored(hex(palette::TextDim), "%s (unsupported type)", param.name.c_str());
                                        break;
                                }

                                ImGui::PopID();
                            }
                        }
                    }
                }


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
                    ImGui::TextColored(hex(palette::TextMuted), "SSBO size");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(palette::Text), "%s (x2)", formatSize(pa.size).c_str());

                    ImGui::TextColored(hex(palette::TextMuted), "Stride");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(palette::Text), "%u bytes", pa.elementStride);

                    ImGui::TextColored(hex(palette::TextMuted), "Precision");
                    ImGui::SameLine(110);
                    ImGui::TextColored(hex(palette::Text), "float32");
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
                ImGui::TextColored(hex(palette::TextDim), "Cell size: %.2f", grid->size / grid->cellCount);
            }
        }


        // ── Light ──
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
            }
        }

        // ── Mesh Renderer ──
        auto* meshComp = registry->try_get<MeshRendererComponent>(selectedEntity);
        if (meshComp) {
            if (ImGui::CollapsingHeader("Mesh Renderer", ImGuiTreeNodeFlags_DefaultOpen)) {
                namespace fs = std::filesystem;

                // Mesh file slot — drag-drop target for KMRB_MESH_PATH
                {
                    std::string meshLabel = meshDisplayLabel(meshComp->meshPath);
                    ImGui::TextColored(hex(palette::TextMuted), "  Mesh");
                    ImGui::SameLine(110);
                    ImGui::TextColored(meshComp->meshPath.empty() ? hex(palette::TextDim) : hex(palette::Text),
                                       "%s", meshLabel.c_str());

                    if (!meshComp->meshPath.empty()) {
                        ImGui::SameLine();
                        if (ImGui::SmallButton("x##clear_mesh")) {
                            meshComp->meshPath.clear();
                            meshComp->meshCacheKey.clear();
                            meshComp->shaderDirty = true;
                        }
                    }

                    if (ImGui::BeginDragDropTarget()) {
                        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("KMRB_MESH_PATH")) {
                            std::string droppedPath(static_cast<const char*>(payload->Data));
                            meshComp->meshPath = droppedPath;
                            meshComp->meshCacheKey.clear(); // Force reload
                            meshComp->shaderDirty = true;
                            Log::ok("Mesh set: " + fs::path(droppedPath).filename().string());
                        }
                        ImGui::EndDragDropTarget();
                    }
                }

                // ── Geometry stats ──
                if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen)) {
                    if (meshComp->vertexCount == 0) {
                        ImGui::TextColored(hex(palette::TextDim), "  Loading...");
                    } else {
                        auto fmtBytes = [](uint32_t b) -> std::string {
                            if (b >= 1024 * 1024)
                                return std::to_string(b / (1024 * 1024)) + "." +
                                       std::to_string((b % (1024 * 1024)) * 10 / (1024 * 1024)) + " MB";
                            if (b >= 1024)
                                return std::to_string(b / 1024) + "." +
                                       std::to_string((b % 1024) * 10 / 1024) + " KB";
                            return std::to_string(b) + " B";
                        };

                        ImGui::TextColored(hex(palette::TextMuted), "  Vertices");
                        ImGui::SameLine(110);
                        ImGui::TextColored(hex(palette::Text), "%u", meshComp->vertexCount);

                        ImGui::TextColored(hex(palette::TextMuted), "  Triangles");
                        ImGui::SameLine(110);
                        ImGui::TextColored(hex(palette::Text), "%u", meshComp->indexCount / 3);

                        ImGui::TextColored(hex(palette::TextMuted), "  GPU Mem");
                        ImGui::SameLine(110);
                        ImGui::TextColored(hex(palette::Text), "%s",
                            meshComp->gpuBytes > 0 ? fmtBytes(meshComp->gpuBytes).c_str() : "—");

                        ImGui::TextColored(hex(palette::TextMuted), "  Status");
                        ImGui::SameLine(110);
                        if (meshInstancesPtr) {
                            auto* insts = static_cast<std::unordered_map<uint32_t,
                                              Renderer::MeshShaderInstance>*>(meshInstancesPtr);
                            uint32_t ek = static_cast<uint32_t>(selectedEntity);
                            bool ready = insts->count(ek) && (*insts).at(ek).graphicsPipeline;
                            if (ready) ImGui::TextColored(hex(palette::Green), "Ready");
                            else       ImGui::TextColored(hex(palette::Gold), "Building");
                        } else {
                            ImGui::TextColored(hex(palette::TextDim), "—");
                        }
                    }
                }

                // Shader slots (Vertex + Fragment) — same drag-drop pattern as Pipeline
                auto drawMeshShaderSlot = [&](const char* label, std::string& path, const char* ext) {
                    std::string filename = path.empty() ? "(default)" : fs::path(path).filename().string();
                    ImGui::TextColored(hex(palette::TextMuted), "  %s", label);
                    ImGui::SameLine(110);
                    ImGui::TextColored(path.empty() ? hex(palette::TextDim) : hex(palette::Text), "%s", filename.c_str());

                    if (!path.empty()) {
                        ImGui::SameLine();
                        std::string clearId = std::string("x##clear_mesh_") + label;
                        if (ImGui::SmallButton(clearId.c_str())) {
                            path.clear();
                            meshComp->shaderDirty = true;
                        }
                    }

                    if (ImGui::BeginDragDropTarget()) {
                        if (const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("KMRB_SHADER_PATH")) {
                            std::string droppedPath(static_cast<const char*>(payload->Data));
                            std::string droppedExt = fs::path(droppedPath).extension().string();
                            if (droppedExt == ext) {
                                path = droppedPath;
                                meshComp->shaderDirty = true;
                                Log::ok("Attached: " + fs::path(droppedPath).filename().string());
                            } else {
                                Log::warn("Wrong shader type for this slot");
                            }
                        }
                        ImGui::EndDragDropTarget();
                    }
                };

                ImGui::Spacing();
                ImGui::TextColored(hex(palette::Gold), "Shaders");
                drawMeshShaderSlot("Vertex", meshComp->vertexShaderPath, ".vert");
                drawMeshShaderSlot("Fragment", meshComp->fragmentShaderPath, ".frag");

                ImGui::Spacing();
                ImGui::TextColored(hex(palette::Gold), "Material");
                ImGui::ColorEdit4("Color", &meshComp->color.x);
                bool wireChanged = ImGui::Checkbox("Wireframe", &meshComp->wireframe);
                if (wireChanged) meshComp->shaderDirty = true;

                // Reflected push constant parameters (user-defined shader params)
                if (meshInstancesPtr) {
                    auto* instances = static_cast<std::unordered_map<uint32_t, Renderer::MeshShaderInstance>*>(meshInstancesPtr);
                    uint32_t key = static_cast<uint32_t>(selectedEntity);
                    auto instIt = instances->find(key);
                    if (instIt != instances->end() && !instIt->second.reflectedParams.empty()) {
                        ImGui::Spacing();
                        ImGui::TextColored(hex(palette::Gold), "Parameters");
                        auto& inst = instIt->second;
                        for (auto& param : inst.reflectedParams) {
                            ImGui::PushID(param.offset);
                            void* ptr = inst.pushConstantData.data() + param.offset;
                            switch (param.type) {
                                case Renderer::ReflectedParam::Float:
                                    ImGui::DragFloat(param.name.c_str(), (float*)ptr, 0.01f); break;
                                case Renderer::ReflectedParam::Vec2:
                                    ImGui::DragFloat2(param.name.c_str(), (float*)ptr, 0.01f); break;
                                case Renderer::ReflectedParam::Vec3:
                                    ImGui::DragFloat3(param.name.c_str(), (float*)ptr, 0.01f); break;
                                case Renderer::ReflectedParam::Vec4:
                                    ImGui::DragFloat4(param.name.c_str(), (float*)ptr, 0.01f); break;
                                case Renderer::ReflectedParam::Int:
                                    ImGui::DragInt(param.name.c_str(), (int*)ptr); break;
                                case Renderer::ReflectedParam::Bool:
                                    ImGui::Checkbox(param.name.c_str(), (bool*)ptr); break;
                                default: break;
                            }
                            ImGui::PopID();
                        }
                    }
                }
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
        ImGui::TextColored(hex(palette::TextDim), "(%zu entries)", Log::getEntries().size());
        ImGui::Separator();

        // Scrollable log
        ImGui::BeginChild("log_scroll", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

        for (auto& entry : Log::getEntries()) {
            // Timestamp
            int mins = static_cast<int>(entry.timestamp) / 60;
            float secs = entry.timestamp - mins * 60;
            ImGui::TextColored(hex(palette::TextDim), "%02d:%05.2f", mins, secs);
            ImGui::SameLine();

            // Level tag with color
            switch (entry.level) {
                case LogLevel::Info:
                    ImGui::TextColored(hex(palette::Blue), "[INF]"); break;
                case LogLevel::Ok:
                    ImGui::TextColored(hex(palette::Green), "[OK]");  break;
                case LogLevel::Warn:
                    ImGui::TextColored(hex(palette::Gold), "[WRN]"); break;
                case LogLevel::Error:
                    ImGui::TextColored(hex(palette::Red), "[ERR]"); break;
            }
            ImGui::SameLine();

            // Message — errors in red, rest in primary text
            if (entry.level == LogLevel::Error)
                ImGui::TextColored(hex(palette::Red), "%s", entry.message.c_str());
            else
                ImGui::TextColored(hex(palette::Text), "%s", entry.message.c_str());
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
                        // latestParticleBuffer is ping-pong-aware (set by the
                        // renderer) — a fixed name would read one step stale
                        if (bufferManager->exists(latestParticleBuffer)) {
                            cachedParticleData = bufferManager->readBack(latestParticleBuffer);
                            cachedElementCount = bufferManager->getInfo(latestParticleBuffer).elementCount;
                        }
                    }
                }

                ImGui::Separator();

                if (cachedParticleData.empty()) {
                    ImGui::TextColored(hex(palette::TextDim), "No data — waiting for first refresh...");
                } else {
                    ImGui::TextColored(hex(palette::TextDim), "%u particles", cachedElementCount);

                    // Scrollable particle data table. Float layout follows the
                    // Particle struct in kmrb_types.hpp (3 vec4s):
                    //   [0] pos.x  [1] pos.y  [2] pos.z  [3] pointSize
                    //   [4] vel.x  [5] vel.y  [6] vel.z  [7] lifetime
                    //   [8] r      [9] g      [10] b     [11] a
                    constexpr int FLOATS_PER_PARTICLE = sizeof(Particle) / sizeof(float);

                    ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollY
                                               | ImGuiTableFlags_RowBg
                                               | ImGuiTableFlags_BordersOuter
                                               | ImGuiTableFlags_BordersV
                                               | ImGuiTableFlags_Resizable
                                               | ImGuiTableFlags_Reorderable;

                    if (ImGui::BeginTable("particle_table", 10, tableFlags)) {
                        ImGui::TableSetupScrollFreeze(0, 1); // Freeze header row
                        ImGui::TableSetupColumn("ID",       ImGuiTableColumnFlags_WidthFixed, 50.0f);
                        ImGui::TableSetupColumn("pos.x",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("pos.y",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("pos.z",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.x",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.y",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("vel.z",    ImGuiTableColumnFlags_WidthFixed, 70.0f);
                        ImGui::TableSetupColumn("size",     ImGuiTableColumnFlags_WidthFixed, 55.0f);
                        ImGui::TableSetupColumn("lifetime", ImGuiTableColumnFlags_WidthFixed, 65.0f);
                        ImGui::TableSetupColumn("color",    ImGuiTableColumnFlags_WidthFixed, 50.0f);
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
                                ImGui::Text("%.2f", cachedParticleData[offset + 3]); // position.w = point size
                                ImGui::TableSetColumnIndex(8);
                                ImGui::Text("%.2f", cachedParticleData[offset + 7]); // velocity.w = lifetime
                                ImGui::TableSetColumnIndex(9);
                                // Color swatch — hover shows the exact RGBA values
                                ImVec4 pcol(cachedParticleData[offset + 8],  cachedParticleData[offset + 9],
                                            cachedParticleData[offset + 10], cachedParticleData[offset + 11]);
                                ImGui::PushID(row);
                                ImGui::ColorButton("##pcol", pcol, ImGuiColorEditFlags_AlphaPreview, ImVec2(36, 14));
                                ImGui::PopID();
                            }
                        }
                        ImGui::EndTable();
                    }
                }
                ImGui::EndTabItem();
            }

            // ── Mesh Stats tab ──
            if (ImGui::BeginTabItem("Mesh")) {
                auto formatMeshSize = [](uint32_t bytes) -> std::string {
                    if (bytes >= 1024 * 1024)
                        return std::to_string(bytes / (1024 * 1024)) + "." +
                               std::to_string((bytes % (1024 * 1024)) * 10 / (1024 * 1024)) + " MB";
                    if (bytes >= 1024)
                        return std::to_string(bytes / 1024) + "." +
                               std::to_string((bytes % 1024) * 10 / 1024) + " KB";
                    return std::to_string(bytes) + " B";
                };

                if (!registry) {
                    ImGui::TextColored(hex(palette::TextDim), "No scene loaded");
                } else {
                    auto meshView = registry->view<MeshRendererComponent, Name>();

                    // Aggregate totals
                    uint32_t totalVerts = 0, totalTris = 0, totalBytes = 0;
                    int meshCount = 0;
                    for (auto entity : meshView) {
                        auto& m = meshView.get<MeshRendererComponent>(entity);
                        totalVerts += m.vertexCount;
                        totalTris  += m.indexCount / 3;
                        totalBytes += m.gpuBytes;
                        meshCount++;
                    }

                    // Summary line
                    ImGui::TextColored(hex(palette::TextMuted), "Total");
                    ImGui::SameLine(60);
                    ImGui::TextColored(hex(palette::Text), "%d mesh%s", meshCount, meshCount == 1 ? "" : "es");
                    ImGui::SameLine(0, 20);
                    ImGui::TextColored(hex(palette::TextMuted), "Verts");
                    ImGui::SameLine();
                    ImGui::TextColored(hex(palette::Text), "%u", totalVerts);
                    ImGui::SameLine(0, 20);
                    ImGui::TextColored(hex(palette::TextMuted), "Tris");
                    ImGui::SameLine();
                    ImGui::TextColored(hex(palette::Text), "%u", totalTris);
                    ImGui::SameLine(0, 20);
                    ImGui::TextColored(hex(palette::TextMuted), "GPU");
                    ImGui::SameLine();
                    ImGui::TextColored(hex(palette::Text), "%s", formatMeshSize(totalBytes).c_str());

                    ImGui::Separator();

                    if (meshCount == 0) {
                        ImGui::TextColored(hex(palette::TextDim), "No mesh entities — add a Mesh in the Scene Hierarchy");
                    } else {
                        ImGuiTableFlags tableFlags = ImGuiTableFlags_ScrollY
                                                   | ImGuiTableFlags_RowBg
                                                   | ImGuiTableFlags_BordersOuter
                                                   | ImGuiTableFlags_BordersV
                                                   | ImGuiTableFlags_Resizable;

                        if (ImGui::BeginTable("mesh_stats_table", 6, tableFlags)) {
                            ImGui::TableSetupScrollFreeze(0, 1);
                            ImGui::TableSetupColumn("Name",    ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableSetupColumn("File",    ImGuiTableColumnFlags_WidthStretch);
                            ImGui::TableSetupColumn("Verts",   ImGuiTableColumnFlags_WidthFixed, 65.0f);
                            ImGui::TableSetupColumn("Tris",    ImGuiTableColumnFlags_WidthFixed, 65.0f);
                            ImGui::TableSetupColumn("GPU Mem", ImGuiTableColumnFlags_WidthFixed, 72.0f);
                            ImGui::TableSetupColumn("Status",  ImGuiTableColumnFlags_WidthFixed, 62.0f);
                            ImGui::TableHeadersRow();

                            auto* meshInsts = meshInstancesPtr
                                ? static_cast<std::unordered_map<uint32_t, Renderer::MeshShaderInstance>*>(meshInstancesPtr)
                                : nullptr;

                            for (auto entity : meshView) {
                                auto& m  = meshView.get<MeshRendererComponent>(entity);
                                auto* nm = registry->try_get<Name>(entity);

                                ImGui::TableNextRow();

                                ImGui::TableSetColumnIndex(0);
                                ImGui::TextColored(hex(palette::Text), "%s", nm ? nm->value.c_str() : "Mesh");

                                ImGui::TableSetColumnIndex(1);
                                if (m.meshPath.empty()) {
                                    ImGui::TextColored(hex(palette::TextDim), "(no mesh)");
                                } else {
                                    ImGui::TextColored(hex(palette::Tan), "%s",
                                                       meshDisplayLabel(m.meshPath).c_str());
                                }

                                ImGui::TableSetColumnIndex(2);
                                if (m.vertexCount > 0) ImGui::Text("%u", m.vertexCount);
                                else ImGui::TextColored(hex(palette::TextDim), "—");

                                ImGui::TableSetColumnIndex(3);
                                if (m.indexCount > 0) ImGui::Text("%u", m.indexCount / 3);
                                else ImGui::TextColored(hex(palette::TextDim), "—");

                                ImGui::TableSetColumnIndex(4);
                                if (m.gpuBytes > 0) ImGui::Text("%s", formatMeshSize(m.gpuBytes).c_str());
                                else ImGui::TextColored(hex(palette::TextDim), "—");

                                ImGui::TableSetColumnIndex(5);
                                uint32_t ek = static_cast<uint32_t>(entity);
                                bool ready = meshInsts && meshInsts->count(ek) &&
                                             (*meshInsts).at(ek).graphicsPipeline;
                                if (ready) ImGui::TextColored(hex(palette::Green), "Ready");
                                else       ImGui::TextColored(hex(palette::Gold), "Building");
                            }
                            ImGui::EndTable();
                        }
                    }
                }
                ImGui::EndTabItem();
            }

            // ── Export tab ──
            if (ImGui::BeginTabItem("Export")) {
                ImGui::Spacing();

                // Export path with browse button
                ImGui::TextColored(hex(palette::TextMuted), "Export Path");
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

                // Sim frame readout — file names use this number
                ImGui::TextColored(hex(palette::TextMuted), "Sim Frame");
                ImGui::SameLine();
                ImGui::TextColored(hex(palette::Text), "%llu", static_cast<unsigned long long>(simFrame));
                ImGui::TextColored(hex(palette::TextDim), "  Frames = compute steps since Restart (0 = init state).");
                ImGui::TextColored(hex(palette::TextDim), "  Recording writes one CSV per sim frame: name_0042.csv, ...");
                ImGui::TextColored(hex(palette::TextDim), "  Pause pauses the recording; Step captures single frames.");

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                // Prompt for a path if none is set; returns false if the user cancels
                auto ensureExportPath = [&]() -> bool {
                    if (exportPath.empty()) {
                        std::string path = saveFileDialog("CSV File (*.csv)\0*.csv\0", "Export Particles");
                        if (path.empty()) return false;
                        exportPath = path;
                    }
                    return true;
                };

                if (ImGui::Button("Export Current Frame", ImVec2(160, 0))) {
                    if (ensureExportPath() && onExportCSV) onExportCSV(exportPath);
                }
                ImGui::SameLine();

                if (!recordingActive) {
                    if (ImGui::Button("Start Recording", ImVec2(140, 0))) {
                        if (ensureExportPath()) {
                            recordedFrames = 0;
                            recordingActive = true;
                        }
                    }
                } else {
                    // Red stop button + live counter — recording state is always visible
                    ImGui::PushStyleColor(ImGuiCol_Button,        hex(palette::RedDark));
                    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hex(palette::Red));
                    ImGui::PushStyleColor(ImGuiCol_ButtonActive,  hex(palette::RedDarker));
                    if (ImGui::Button("Stop Recording", ImVec2(140, 0))) {
                        recordingActive = false;
                    }
                    ImGui::PopStyleColor(3);
                    ImGui::SameLine();
                    ImGui::TextColored(hex(palette::Red), "Recording — %u frame(s) captured", recordedFrames);
                }

                ImGui::EndTabItem();
            }

            ImGui::EndTabBar();
        }
    }
    ImGui::End();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PREFERENCES WINDOW
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void UI::drawPreferences() {
    // Centered, non-docked window — behaves like a modal settings dialog
    ImGui::SetNextWindowSize(ImVec2(450, 350), ImGuiCond_FirstUseEver);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoCollapse;

    if (ImGui::Begin("Preferences", &showPreferences, flags)) {

        // ── Rendering ──
        if (ImGui::CollapsingHeader("Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(hex(palette::TextMuted), "Resolution");
            ImGui::SameLine(160);
            ImGui::SetNextItemWidth(80);
            if (ImGui::InputInt("##res_w", &renderWidth, 0, 0)) renderResDirty = true;
            ImGui::SameLine();
            ImGui::TextColored(hex(palette::TextDim), "x");
            ImGui::SameLine();
            ImGui::SetNextItemWidth(80);
            if (ImGui::InputInt("##res_h", &renderHeight, 0, 0)) renderResDirty = true;
            renderWidth = std::clamp(renderWidth, 320, 7680);
            renderHeight = std::clamp(renderHeight, 240, 4320);
            ImGui::TextColored(hex(palette::TextDim), "  Offscreen framebuffer size. Affects GPU load.");
        }

        ImGui::Spacing();

        // ── Mesh Rendering ──
        if (ImGui::CollapsingHeader("Mesh Rendering", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(hex(palette::TextMuted), "Default Shader");
            ImGui::SameLine(160);
            ImGui::TextColored(hex(palette::TextDim), "Unlit (engine default)");

            ImGui::Spacing();
            ImGui::TextColored(hex(palette::TextDim), "  Per-mesh shader overrides in Inspector > Shaders.");
            ImGui::TextColored(hex(palette::TextDim), "  PBR / shadow settings coming in V2.");
        }

        ImGui::Spacing();

        // ── Camera ──
        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(hex(palette::TextMuted), "Move Speed");
            ImGui::SameLine(160);
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##cam_speed", &cameraMoveSpeed, 0.5f, 50.0f, "%.1f");

            ImGui::TextColored(hex(palette::TextMuted), "Look Sensitivity");
            ImGui::SameLine(160);
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##cam_sens", &cameraLookSensitivity, 0.01f, 0.5f, "%.2f");
        }

        ImGui::Spacing();

        // ── Viewport ──
        if (ImGui::CollapsingHeader("Viewport", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(hex(palette::TextMuted), "Light Gizmos");
            ImGui::SameLine(160);
            ImGui::Checkbox("##show_gizmos", &showGizmos);
            ImGui::SameLine();
            ImGui::TextColored(hex(palette::TextDim), "Show light position/direction in viewport");
        }

        ImGui::Spacing();

        // ── Data ──
        if (ImGui::CollapsingHeader("Data Output", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(hex(palette::TextMuted), "Refresh Interval");
            ImGui::SameLine(160);
            ImGui::SetNextItemWidth(-1);
            ImGui::SliderFloat("##data_refresh", &dataRefreshInterval, 0.1f, 5.0f, "%.1f s");

            ImGui::Checkbox("Auto-refresh Buffer Table", &dataAutoRefresh);
        }
    }
    ImGui::End();
}

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
    f << "  \"precision\": \"float32\"\n";
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
