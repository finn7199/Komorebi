#include "kmrb_ui.hpp"
#include "kmrb_buffers.hpp"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace kmrb {

void BufferManager::init(vk::Device dev, vk::PhysicalDevice gpu) {
    device = dev;
    physicalDevice = gpu;
}

void BufferManager::cleanup() {
    for (auto& [name, info] : buffers) {
        if (info.mapped) {
            device.unmapMemory(info.memory);
        }
        device.destroyBuffer(info.buffer);
        device.freeMemory(info.memory);
    }
    buffers.clear();
}

void BufferManager::createBuffer(const std::string& name,
                                 vk::DeviceSize size,
                                 vk::BufferUsageFlags usage,
                                 vk::MemoryPropertyFlags memoryProperties,
                                 bool persistentMap) {
    if (exists(name)) {
        destroyBuffer(name);
    }

    BufferInfo info{};
    info.name = name;
    info.size = size;
    info.usage = usage;

    info.buffer = device.createBuffer(vk::BufferCreateInfo({}, size, usage));

    vk::MemoryRequirements memReqs = device.getBufferMemoryRequirements(info.buffer);
    info.memory = device.allocateMemory(vk::MemoryAllocateInfo(
        memReqs.size, findMemoryType(memReqs.memoryTypeBits, memoryProperties)
    ));
    device.bindBufferMemory(info.buffer, info.memory, 0);

    if (persistentMap) {
        info.mapped = device.mapMemory(info.memory, 0, size);
    }

    buffers[name] = info;
}

void BufferManager::createBufferWithData(const std::string& name,
                                         const void* data,
                                         vk::DeviceSize size,
                                         vk::BufferUsageFlags usage,
                                         vk::MemoryPropertyFlags memoryProperties) {
    createBuffer(name, size, usage, memoryProperties, false);

    void* mapped = device.mapMemory(buffers[name].memory, 0, size);
    memcpy(mapped, data, size);
    device.unmapMemory(buffers[name].memory);
}

void BufferManager::upload(const std::string& name, const void* data, vk::DeviceSize size) {
    auto& info = getInfo(name);
    if (info.mapped) {
        memcpy(info.mapped, data, size);
    } else {
        void* mapped = device.mapMemory(info.memory, 0, size);
        memcpy(mapped, data, size);
        device.unmapMemory(info.memory);
    }
}

void BufferManager::setElementInfo(const std::string& name, uint32_t count, uint32_t stride) {
    auto& info = getInfo(name);
    info.elementCount = count;
    info.elementStride = stride;
}

void BufferManager::destroyBuffer(const std::string& name) {
    auto it = buffers.find(name);
    if (it == buffers.end()) return;

    auto& info = it->second;
    if (info.mapped) {
        device.unmapMemory(info.memory);
    }
    device.destroyBuffer(info.buffer);
    device.freeMemory(info.memory);
    buffers.erase(it);
}

vk::Buffer BufferManager::getBuffer(const std::string& name) const {
    return getInfo(name).buffer;
}

BufferInfo& BufferManager::getInfo(const std::string& name) {
    auto it = buffers.find(name);
    if (it == buffers.end()) {
        throw std::runtime_error("KMRB BufferManager: buffer '" + name + "' not found");
    }
    return it->second;
}

const BufferInfo& BufferManager::getInfo(const std::string& name) const {
    auto it = buffers.find(name);
    if (it == buffers.end()) {
        throw std::runtime_error("KMRB BufferManager: buffer '" + name + "' not found");
    }
    return it->second;
}

void* BufferManager::getMappedData(const std::string& name) const {
    return getInfo(name).mapped;
}

bool BufferManager::exists(const std::string& name) const {
    return buffers.count(name) > 0;
}

std::vector<float> BufferManager::readBack(const std::string& name) {
    auto& info = getInfo(name);
    uint32_t floatCount = static_cast<uint32_t>(info.size / sizeof(float));
    std::vector<float> data(floatCount);

    if (info.mapped) {
        // Already mapped — just copy
        memcpy(data.data(), info.mapped, info.size);
    } else {
        // Map, copy, unmap
        void* mapped = device.mapMemory(info.memory, 0, info.size);
        memcpy(data.data(), mapped, info.size);
        device.unmapMemory(info.memory);
    }

    return data;
}

bool BufferManager::exportToCSV(const std::string& bufferName,
                                const std::string& filePath,
                                const std::vector<std::string>& columnHeaders) {
    if (!exists(bufferName)) {
        kmrb::Log::error("Export failed: buffer '" + bufferName + "' not found");
        return false;
    }

    auto& info = getInfo(bufferName);
    if (info.elementCount == 0 || info.elementStride == 0) {
        kmrb::Log::error("Export failed: buffer '" + bufferName + "' has no element info set");
        return false;
    }

    uint32_t floatsPerElement = info.elementStride / sizeof(float);
    std::vector<float> data = readBack(bufferName);

    std::ofstream file(filePath);
    if (!file.is_open()) {
        kmrb::Log::error("Export failed: cannot open " + filePath);
        return false;
    }

    // Write header row
    file << "id";
    if (!columnHeaders.empty()) {
        for (const auto& col : columnHeaders) {
            file << "," << col;
        }
    } else {
        for (uint32_t c = 0; c < floatsPerElement; c++) {
            file << ",col_" << c;
        }
    }
    file << "\n";

    // Write data rows
    for (uint32_t row = 0; row < info.elementCount; row++) {
        file << row;
        uint32_t offset = row * floatsPerElement;
        for (uint32_t c = 0; c < floatsPerElement; c++) {
            file << "," << data[offset + c];
        }
        file << "\n";
    }

    file.close();
    kmrb::Log::ok("Exported " + std::to_string(info.elementCount) + " elements from '" + bufferName + "' to " + filePath);
    return true;
}

uint32_t BufferManager::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProps = physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("KMRB: Failed to find suitable memory type!");
}

} // namespace kmrb
