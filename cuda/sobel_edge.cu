#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kSobelX[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f,
};

__device__ __constant__ float kSobelXDevice[9] = {
    -1.0f, 0.0f, 1.0f,
    -2.0f, 0.0f, 2.0f,
    -1.0f, 0.0f, 1.0f,
};

constexpr float kSobelY[9] = {
    -1.0f, -2.0f, -1.0f,
    0.0f, 0.0f, 0.0f,
    1.0f, 2.0f, 1.0f,
};

__device__ __constant__ float kSobelYDevice[9] = {
    -1.0f, -2.0f, -1.0f,
    0.0f, 0.0f, 0.0f,
    1.0f, 2.0f, 1.0f,
};

void sobel_once_cpu(const unsigned char* source, unsigned char* destination, int width, int height, float strength) {
    const float gain = std::max(1.0f, strength) * 1.6f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float gx = 0.0f;
            float gy = 0.0f;

            for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
                    const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
                    const float gray = cudaimg::grayscale_from_rgba(source, sample_x, sample_y, width, height);
                    const int kernel_index = (kernel_y + 1) * 3 + (kernel_x + 1);
                    gx += gray * kSobelX[kernel_index];
                    gy += gray * kSobelY[kernel_index];
                }
            }

            const int output_index = cudaimg::pixel_offset(x, y, width);
            const unsigned char value = cudaimg::clamp_byte(std::sqrt(gx * gx + gy * gy) * gain);
            destination[output_index + 0] = value;
            destination[output_index + 1] = value;
            destination[output_index + 2] = value;
            destination[output_index + 3] = source[output_index + 3];
        }
    }
}

__global__ void sobel_naive_kernel(const unsigned char* input, unsigned char* output, int width, int height, float strength) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float gx = 0.0f;
    float gy = 0.0f;

    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
            const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
            const float gray = cudaimg::grayscale_from_rgba(input, sample_x, sample_y, width, height);
            const int kernel_index = (kernel_y + 1) * 3 + (kernel_x + 1);
            gx += gray * kSobelXDevice[kernel_index];
            gy += gray * kSobelYDevice[kernel_index];
        }
    }

    const int index = cudaimg::pixel_offset(x, y, width);
    const unsigned char value = cudaimg::clamp_byte(sqrtf(gx * gx + gy * gy) * fmaxf(1.0f, strength) * 1.6f);
    output[index + 0] = value;
    output[index + 1] = value;
    output[index + 2] = value;
    output[index + 3] = input[index + 3];
}

__global__ void sobel_shared_kernel(const unsigned char* input, unsigned char* output, int width, int height, float strength) {
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

    float gx = 0.0f;
    float gy = 0.0f;
    const int local_x = threadIdx.x + 1;
    const int local_y = threadIdx.y + 1;

    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int tile_index = (local_y + kernel_y) * tile_width + (local_x + kernel_x);
            const int kernel_index = (kernel_y + 1) * 3 + (kernel_x + 1);
            gx += tile[tile_index] * kSobelXDevice[kernel_index];
            gy += tile[tile_index] * kSobelYDevice[kernel_index];
        }
    }

    const int index = cudaimg::pixel_offset(x, y, width);
    const unsigned char value = cudaimg::clamp_byte(sqrtf(gx * gx + gy * gy) * fmaxf(1.0f, strength) * 1.6f);
    output[index + 0] = value;
    output[index + 1] = value;
    output[index + 2] = value;
    output[index + 3] = input[index + 3];
}

int run_sobel_cpu_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    sobel_once_cpu(input_rgba, output_rgba, width, height, strength);
    return 0;
}

int run_sobel_cuda_naive_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [strength](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            sobel_naive_kernel<<<grid, block>>>(buffers.input, buffers.output, launch_width, launch_height, strength);
            if (cudaGetLastError() != cudaSuccess) {
                return -1;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return -1;
            }
            return 0;
        });
}

int run_sobel_cuda_optimized_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [strength](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            sobel_shared_kernel<<<grid, block>>>(buffers.input, buffers.output, launch_width, launch_height, strength);
            if (cudaGetLastError() != cudaSuccess) {
                return -1;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return -1;
            }
            return 0;
        });
}

}  // namespace

extern "C" CUDA_IMAGE_API int sobel_edge_cpu(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sobel_cpu_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int sobel_edge_cuda_naive(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sobel_cuda_naive_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int sobel_edge_cuda_optimized(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sobel_cuda_optimized_impl(input_rgba, output_rgba, width, height, strength);
}
