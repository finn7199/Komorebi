#pragma once

#include <vulkan/vulkan.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace kmrb {

// Metadata for a managed buffer — exposed to UI for the Buffer Table panel
struct BufferInfo {
    std::string name;
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::DeviceSize size = 0;
    vk::BufferUsageFlags usage;
    void* mapped = nullptr;         // Non-null if persistently mapped
    uint32_t elementCount = 0;      // Number of structs (for display)
    uint32_t elementStride = 0;     // Sizeof one element (for display)
};

class BufferManager {
public:
    void init(vk::Device device, vk::PhysicalDevice physicalDevice);
    void cleanup();

    // Create a named buffer. Returns a handle (the name) for later access.
    // If persistentMap is true, the buffer stays mapped for CPU writes.
    void createBuffer(const std::string& name,
                      vk::DeviceSize size,
                      vk::BufferUsageFlags usage,
                      vk::MemoryPropertyFlags memoryProperties,
                      bool persistentMap = false);

    // Convenience: create + upload initial data
    void createBufferWithData(const std::string& name,
                              const void* data,
                              vk::DeviceSize size,
                              vk::BufferUsageFlags usage,
                              vk::MemoryPropertyFlags memoryProperties);

    // Upload data to an existing mapped buffer
    void upload(const std::string& name, const void* data, vk::DeviceSize size);

    // Set element info for UI display (e.g., "10000 particles, 48 bytes each")
    void setElementInfo(const std::string& name, uint32_t count, uint32_t stride);

    // Destroy a single buffer
    void destroyBuffer(const std::string& name);

    // Access
    vk::Buffer getBuffer(const std::string& name) const;
    BufferInfo& getInfo(const std::string& name);
    const BufferInfo& getInfo(const std::string& name) const;
    void* getMappedData(const std::string& name) const;
    bool exists(const std::string& name) const;

    // Read back buffer data to CPU (maps, copies, unmaps if not persistently mapped)
    std::vector<float> readBack(const std::string& name);

    // Export any buffer to CSV — interprets data as rows of floats
    // Column headers are optional. If empty, generates "col_0", "col_1", etc.
    // Works with any SSBO: particles, lights, vertices, BRDF data — all are packed floats.
    bool exportToCSV(const std::string& bufferName,
                     const std::string& filePath,
                     const std::vector<std::string>& columnHeaders = {});

    // For UI iteration (Buffer Table panel)
    const std::unordered_map<std::string, BufferInfo>& getAllBuffers() const { return buffers; }

private:
    vk::Device device;
    vk::PhysicalDevice physicalDevice;
    std::unordered_map<std::string, BufferInfo> buffers;

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
};

} // namespace kmrb
