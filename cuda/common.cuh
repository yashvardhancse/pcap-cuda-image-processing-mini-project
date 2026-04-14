#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(_WIN32)
    #define CUDA_IMAGE_API __declspec(dllexport)
#else
    #define CUDA_IMAGE_API
#endif

namespace cudaimg {

constexpr int kChannels = 4;
constexpr int kTileSize = 16;

struct DeviceBuffers {
    unsigned char* input = nullptr;
    unsigned char* output = nullptr;
    unsigned char* scratch = nullptr;
};

__host__ __device__ inline int clamp_int(int value, int minimum, int maximum) {
    return value < minimum ? minimum : (value > maximum ? maximum : value);
}

__host__ __device__ inline unsigned char clamp_byte(float value) {
    return static_cast<unsigned char>(value < 0.0f ? 0.0f : (value > 255.0f ? 255.0f : value));
}

__host__ __device__ inline int pixel_offset(int x, int y, int width) {
    return (y * width + x) * kChannels;
}

__host__ __device__ inline float grayscale_from_rgba(const unsigned char* image, int x, int y, int width, int height) {
    x = clamp_int(x, 0, width - 1);
    y = clamp_int(y, 0, height - 1);
    const int offset = pixel_offset(x, y, width);
    return 0.299f * image[offset + 0] + 0.587f * image[offset + 1] + 0.114f * image[offset + 2];
}

inline std::size_t image_bytes(int width, int height) {
    return static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * static_cast<std::size_t>(kChannels);
}

inline dim3 block_dim() {
    return dim3(kTileSize, kTileSize);
}

inline dim3 grid_dim(int width, int height) {
    return dim3((width + kTileSize - 1) / kTileSize, (height + kTileSize - 1) / kTileSize);
}

inline bool allocate(DeviceBuffers& buffers, std::size_t bytes) {
    if (cudaMalloc(reinterpret_cast<void**>(&buffers.input), bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&buffers.output), bytes) != cudaSuccess) {
        cudaFree(buffers.input);
        buffers.input = nullptr;
        return false;
    }
    if (cudaMalloc(reinterpret_cast<void**>(&buffers.scratch), bytes) != cudaSuccess) {
        cudaFree(buffers.output);
        cudaFree(buffers.input);
        buffers.output = nullptr;
        buffers.input = nullptr;
        return false;
    }
    return true;
}

inline void release(DeviceBuffers& buffers) {
    if (buffers.scratch != nullptr) {
        cudaFree(buffers.scratch);
        buffers.scratch = nullptr;
    }
    if (buffers.output != nullptr) {
        cudaFree(buffers.output);
        buffers.output = nullptr;
    }
    if (buffers.input != nullptr) {
        cudaFree(buffers.input);
        buffers.input = nullptr;
    }
}

inline bool copy_host_to_device(const unsigned char* host, unsigned char* device, std::size_t bytes) {
    return cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

inline bool copy_device_to_host(unsigned char* host, const unsigned char* device, std::size_t bytes) {
    return cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

inline bool copy_device_to_device(unsigned char* destination, const unsigned char* source, std::size_t bytes) {
    return cudaMemcpy(destination, source, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
}

template <typename Launcher>
int run_cuda_operation(const unsigned char* input, unsigned char* output, int width, int height, Launcher&& launcher) {
    DeviceBuffers buffers{};
    const std::size_t bytes = image_bytes(width, height);

    if (!allocate(buffers, bytes)) {
        return -1;
    }
    if (!copy_host_to_device(input, buffers.input, bytes)) {
        release(buffers);
        return -1;
    }

    const int launch_result = launcher(buffers, width, height, bytes);
    if (launch_result != 0) {
        release(buffers);
        return launch_result;
    }

    if (!copy_device_to_host(output, buffers.output, bytes)) {
        release(buffers);
        return -1;
    }

    release(buffers);
    return 0;
}

}  // namespace cudaimg
