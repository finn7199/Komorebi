#include "kmrb_mesh.hpp"
#include "kmrb_ui.hpp"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <filesystem>
#include <algorithm>
#include <functional>

namespace kmrb {

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PUBLIC API
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

std::string MeshCache::load(const std::string& filePath, BufferManager& buffers) {
    // Normalize path as cache key (forward slashes, canonical)
    std::string key = std::filesystem::path(filePath).lexically_normal().generic_string();

    if (meshes.count(key)) return key;

    MeshData data = loadFromFile(filePath);
    if (data.vertices.empty()) {
        Log::error("Failed to load mesh: " + filePath);
        return "";
    }

    uploadToGPU(key, data, buffers);
    Log::ok("Loaded mesh: " + std::filesystem::path(filePath).filename().string()
            + " (" + std::to_string(data.vertices.size()) + " verts, "
            + std::to_string(data.indices.size() / 3) + " tris)");
    return key;
}

void MeshCache::loadPrimitives(BufferManager& buffers) {
    uploadToGPU("__primitive_cube", generateCube(), buffers);
    uploadToGPU("__primitive_sphere", generateSphere(), buffers);
    uploadToGPU("__primitive_plane", generatePlane(), buffers);
}

const GPUMesh& MeshCache::get(const std::string& key) const {
    return meshes.at(key);
}

bool MeshCache::exists(const std::string& key) const {
    return meshes.count(key) > 0;
}

void MeshCache::cleanup(BufferManager& buffers) {
    for (auto& [key, mesh] : meshes) {
        if (buffers.exists(mesh.vertexBufferName)) buffers.destroyBuffer(mesh.vertexBufferName);
        if (buffers.exists(mesh.indexBufferName))   buffers.destroyBuffer(mesh.indexBufferName);
    }
    meshes.clear();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// GPU UPLOAD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

void MeshCache::uploadToGPU(const std::string& key, const MeshData& data, BufferManager& buffers) {
    // Generate unique buffer names from a hash of the key
    size_t hash = std::hash<std::string>{}(key);
    std::string vtxName = "mesh_vtx_" + std::to_string(hash);
    std::string idxName = "mesh_idx_" + std::to_string(hash);

    vk::DeviceSize vtxSize = data.vertices.size() * sizeof(MeshVertex);
    vk::DeviceSize idxSize = data.indices.size() * sizeof(uint32_t);

    buffers.createBufferWithData(vtxName, data.vertices.data(), vtxSize,
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    buffers.createBufferWithData(idxName, data.indices.data(), idxSize,
        vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    GPUMesh mesh;
    mesh.sourceFile = key;
    mesh.vertexCount = static_cast<uint32_t>(data.vertices.size());
    mesh.indexCount = static_cast<uint32_t>(data.indices.size());
    mesh.vertexBufferName = vtxName;
    mesh.indexBufferName = idxName;
    meshes[key] = std::move(mesh);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ASSIMP FILE LOADING
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MeshData MeshCache::loadFromFile(const std::string& filePath) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filePath,
        aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);

    if (!scene || !scene->HasMeshes()) {
        Log::error("Assimp: " + std::string(importer.GetErrorString()));
        return {};
    }

    // V1: load the first mesh in the scene
    const aiMesh* mesh = scene->mMeshes[0];
    MeshData result;
    result.vertices.reserve(mesh->mNumVertices);

    for (unsigned int i = 0; i < mesh->mNumVertices; i++) {
        MeshVertex v{};
        v.position = { mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z };

        if (mesh->HasNormals()) {
            v.normal = { mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z };
        }

        if (mesh->HasTextureCoords(0)) {
            v.uv = { mesh->mTextureCoords[0][i].x, mesh->mTextureCoords[0][i].y };
        }

        result.vertices.push_back(v);
    }

    for (unsigned int i = 0; i < mesh->mNumFaces; i++) {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++) {
            result.indices.push_back(face.mIndices[j]);
        }
    }

    return result;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// BUILT-IN PRIMITIVES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MeshData MeshCache::generateCube() {
    MeshData data;

    // 24 vertices (4 per face, unique normals per face)
    struct FaceVerts { glm::vec3 normal; glm::vec3 positions[4]; glm::vec2 uvs[4]; };
    FaceVerts faces[] = {
        // Front  (+Z)
        {{ 0, 0, 1}, {{ -0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
        // Back   (-Z)
        {{ 0, 0,-1}, {{ 0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
        // Right  (+X)
        {{ 1, 0, 0}, {{ 0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
        // Left   (-X)
        {{-1, 0, 0}, {{-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f,-0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
        // Top    (+Y)
        {{ 0, 1, 0}, {{-0.5f, 0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, { 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
        // Bottom (-Y)
        {{ 0,-1, 0}, {{-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f, 0.5f}, {-0.5f,-0.5f, 0.5f}},
         {{0,0}, {1,0}, {1,1}, {0,1}}},
    };

    for (int f = 0; f < 6; f++) {
        uint32_t base = static_cast<uint32_t>(data.vertices.size());
        for (int v = 0; v < 4; v++) {
            data.vertices.push_back({ faces[f].positions[v], faces[f].normal, faces[f].uvs[v] });
        }
        // Two triangles per face
        data.indices.insert(data.indices.end(), { base, base+1, base+2, base, base+2, base+3 });
    }

    return data;
}

MeshData MeshCache::generateSphere(int segments, int rings) {
    MeshData data;

    for (int y = 0; y <= rings; y++) {
        for (int x = 0; x <= segments; x++) {
            float xSeg = static_cast<float>(x) / segments;
            float ySeg = static_cast<float>(y) / rings;
            float theta = xSeg * 2.0f * glm::pi<float>();
            float phi   = ySeg * glm::pi<float>();

            glm::vec3 pos = {
                std::cos(theta) * std::sin(phi),
                std::cos(phi),
                std::sin(theta) * std::sin(phi)
            };

            data.vertices.push_back({ pos * 0.5f, pos, { xSeg, ySeg } });
        }
    }

    for (int y = 0; y < rings; y++) {
        for (int x = 0; x < segments; x++) {
            uint32_t tl = y * (segments + 1) + x;
            uint32_t tr = tl + 1;
            uint32_t bl = tl + (segments + 1);
            uint32_t br = bl + 1;
            data.indices.insert(data.indices.end(), { tl, bl, tr, tr, bl, br });
        }
    }

    return data;
}

MeshData MeshCache::generatePlane() {
    MeshData data;
    glm::vec3 normal = { 0, 1, 0 };

    data.vertices = {
        {{ -0.5f, 0, -0.5f }, normal, { 0, 0 }},
        {{  0.5f, 0, -0.5f }, normal, { 1, 0 }},
        {{  0.5f, 0,  0.5f }, normal, { 1, 1 }},
        {{ -0.5f, 0,  0.5f }, normal, { 0, 1 }},
    };
    data.indices = { 0, 1, 2, 0, 2, 3 };

    return data;
}

} // namespace kmrb
