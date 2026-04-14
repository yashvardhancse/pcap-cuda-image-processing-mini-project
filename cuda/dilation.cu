#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

constexpr bool kDilate = true;

void morphology_once_cpu(const unsigned char* source, unsigned char* destination, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float extreme = kDilate ? 0.0f : 255.0f;
            for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
                    const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
                    const float gray = cudaimg::grayscale_from_rgba(source, sample_x, sample_y, width, height);
                    extreme = kDilate ? std::max(extreme, gray) : std::min(extreme, gray);
                }
            }

            const int index = cudaimg::pixel_offset(x, y, width);
            const unsigned char value = cudaimg::clamp_byte(extreme);
            destination[index + 0] = value;
            destination[index + 1] = value;
            destination[index + 2] = value;
            destination[index + 3] = source[index + 3];
        }
    }
}

__global__ void morphology_naive_kernel(const unsigned char* input, unsigned char* output, int width, int height, int dilate) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float extreme = dilate ? 0.0f : 255.0f;
    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
            const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
            const float gray = cudaimg::grayscale_from_rgba(input, sample_x, sample_y, width, height);
            extreme = dilate ? fmaxf(extreme, gray) : fminf(extreme, gray);
        }
    }

    const int index = cudaimg::pixel_offset(x, y, width);
    const unsigned char value = cudaimg::clamp_byte(extreme);
    output[index + 0] = value;
    output[index + 1] = value;
    output[index + 2] = value;
    output[index + 3] = input[index + 3];
}

__global__ void morphology_shared_kernel(const unsigned char* input, unsigned char* output, int width, int height, int dilate) {
    constexpr int tile_width = cudaimg::kTileSize + 2;
    constexpr int tile_height = cudaimg::kTileSize + 2;
    __shared__ float tile[tile_width * tile_height];

    const int block_x = blockIdx.x * cudaimg::kTileSize;
    const int block_y = blockIdx.y * cudaimg::kTileSize;
    const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    for (int index = thread_index; index < tile_width * tile_height; index += threads) {
        const int local_x = index % tile_width;
        const int local_y = index / tile_width;
        const int global_x = cudaimg::clamp_int(block_x + local_x - 1, 0, width - 1);
        const int global_y = cudaimg::clamp_int(block_y + local_y - 1, 0, height - 1);
        tile[index] = cudaimg::grayscale_from_rgba(input, global_x, global_y, width, height);
    }

    __syncthreads();

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float extreme = dilate ? 0.0f : 255.0f;
    const int local_x = threadIdx.x + 1;
    const int local_y = threadIdx.y + 1;

    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int tile_index = (local_y + kernel_y) * tile_width + (local_x + kernel_x);
            extreme = dilate ? fmaxf(extreme, tile[tile_index]) : fminf(extreme, tile[tile_index]);
        }
    }

    const int index = cudaimg::pixel_offset(x, y, width);
    const unsigned char value = cudaimg::clamp_byte(extreme);
    output[index + 0] = value;
    output[index + 1] = value;
    output[index + 2] = value;
    output[index + 3] = input[index + 3];
}

int run_morphology_cpu_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const int passes = std::max(1, static_cast<int>(std::round(strength)));
    const std::size_t bytes = cudaimg::image_bytes(width, height);
    std::vector<unsigned char> scratch(bytes);
    const unsigned char* source = input_rgba;
    unsigned char* destination = output_rgba;

    for (int pass = 0; pass < passes; ++pass) {
        morphology_once_cpu(source, destination, width, height);
        source = destination;
        destination = (destination == output_rgba) ? scratch.data() : output_rgba;
    }

    if (source != output_rgba) {
        std::memcpy(output_rgba, source, bytes);
    }
    return 0;
}

int run_morphology_cuda_naive_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const int passes = std::max(1, static_cast<int>(std::round(strength)));
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [passes](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            unsigned char* source = buffers.input;
            unsigned char* destination = buffers.output;

            for (int pass = 0; pass < passes; ++pass) {
                morphology_naive_kernel<<<grid, block>>>(source, destination, launch_width, launch_height, kDilate ? 1 : 0);
                if (cudaGetLastError() != cudaSuccess) {
                    return -1;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    return -1;
                }
                std::swap(source, destination);
            }

            if (source != buffers.output && !cudaimg::copy_device_to_device(buffers.output, source, bytes)) {
                return -1;
            }
            return 0;
        });
}

int run_morphology_cuda_optimized_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const int passes = std::max(1, static_cast<int>(std::round(strength)));
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [passes](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            unsigned char* source = buffers.input;
            unsigned char* destination = buffers.output;

            for (int pass = 0; pass < passes; ++pass) {
                morphology_shared_kernel<<<grid, block>>>(source, destination, launch_width, launch_height, kDilate ? 1 : 0);
                if (cudaGetLastError() != cudaSuccess) {
                    return -1;
                }
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    return -1;
                }
                std::swap(source, destination);
            }

            if (source != buffers.output && !cudaimg::copy_device_to_device(buffers.output, source, bytes)) {
                return -1;
            }
            return 0;
        });
}

}  // namespace

extern "C" CUDA_IMAGE_API int dilation_cpu(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_morphology_cpu_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int dilation_cuda_naive(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_morphology_cuda_naive_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int dilation_cuda_optimized(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_morphology_cuda_optimized_impl(input_rgba, output_rgba, width, height, strength);
}
