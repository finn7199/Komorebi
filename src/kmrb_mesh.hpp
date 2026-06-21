#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <vector>
#include <unordered_map>

#include "kmrb_types.hpp"
#include "kmrb_buffers.hpp"

namespace kmrb {

// Built-in primitive cache keys. A primitive choice is stored as a sentinel in
// MeshRendererComponent::meshPath (the serializable field), so it round-trips through
// save/load once that lands. The renderer recognizes PRIMITIVE_PREFIX and pulls the mesh
// straight from the cache instead of trying to load it as a file.
inline constexpr const char* PRIMITIVE_PREFIX = "__primitive_";
inline constexpr const char* PRIMITIVE_CUBE   = "__primitive_cube";
inline constexpr const char* PRIMITIVE_SPHERE = "__primitive_sphere";

// CPU-side mesh data loaded from file (before GPU upload)
struct MeshData {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

// GPU-resident mesh — references named buffers in BufferManager
struct GPUMesh {
    std::string sourceFile;
    uint32_t vertexCount = 0;
    uint32_t indexCount = 0;
    std::string vertexBufferName;   // Key in BufferManager
    std::string indexBufferName;    // Key in BufferManager
};

// Loads 3D models via Assimp, uploads to GPU, and deduplicates by file path.
// Also provides built-in primitives (cube, sphere) for default entities.
class MeshCache {
public:
    // Load a mesh from file and upload to GPU. Returns cache key.
    // If the file was already loaded, returns the existing key (no reload).
    std::string load(const std::string& filePath, BufferManager& buffers);

    // Load built-in primitives (called once at init)
    void loadPrimitives(BufferManager& buffers);

    const GPUMesh& get(const std::string& key) const;
    bool exists(const std::string& key) const;

    // Destroy all GPU buffers
    void cleanup(BufferManager& buffers);

private:
    std::unordered_map<std::string, GPUMesh> meshes;

    // Upload MeshData to GPU and store in cache under the given key
    void uploadToGPU(const std::string& key, const MeshData& data, BufferManager& buffers);

    // Assimp file loading
    static MeshData loadFromFile(const std::string& filePath);

    // Built-in primitive generators
    static MeshData generateCube();
    static MeshData generateSphere(int segments = 32, int rings = 16);
};

} // namespace kmrb
