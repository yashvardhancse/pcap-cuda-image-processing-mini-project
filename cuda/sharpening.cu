#include "common.cuh"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

constexpr float kKernel[9] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f,
};

__device__ __constant__ float kKernelDevice[9] = {
    1.0f, 2.0f, 1.0f,
    2.0f, 4.0f, 2.0f,
    1.0f, 2.0f, 1.0f,
};

void blur_once_cpu(const unsigned char* source, unsigned char* destination, int width, int height) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float red = 0.0f;
            float green = 0.0f;
            float blue = 0.0f;
            const int center = cudaimg::pixel_offset(x, y, width);

            for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
                for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
                    const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
                    const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
                    const int sample = cudaimg::pixel_offset(sample_x, sample_y, width);
                    const float weight = kKernel[(kernel_y + 1) * 3 + (kernel_x + 1)];
                    red += weight * source[sample + 0];
                    green += weight * source[sample + 1];
                    blue += weight * source[sample + 2];
                }
            }

            destination[center + 0] = cudaimg::clamp_byte(red / 16.0f);
            destination[center + 1] = cudaimg::clamp_byte(green / 16.0f);
            destination[center + 2] = cudaimg::clamp_byte(blue / 16.0f);
            destination[center + 3] = source[center + 3];
        }
    }
}

__global__ void sharpen_naive_kernel(const unsigned char* input, unsigned char* output, int width, int height, float amount) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float blurred_red = 0.0f;
    float blurred_green = 0.0f;
    float blurred_blue = 0.0f;
    const int center = cudaimg::pixel_offset(x, y, width);

    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int sample_x = cudaimg::clamp_int(x + kernel_x, 0, width - 1);
            const int sample_y = cudaimg::clamp_int(y + kernel_y, 0, height - 1);
            const int sample = cudaimg::pixel_offset(sample_x, sample_y, width);
            const float weight = kKernelDevice[(kernel_y + 1) * 3 + (kernel_x + 1)];
            blurred_red += weight * input[sample + 0];
            blurred_green += weight * input[sample + 1];
            blurred_blue += weight * input[sample + 2];
        }
    }

    blurred_red /= 16.0f;
    blurred_green /= 16.0f;
    blurred_blue /= 16.0f;

    output[center + 0] = cudaimg::clamp_byte(input[center + 0] + amount * (input[center + 0] - blurred_red));
    output[center + 1] = cudaimg::clamp_byte(input[center + 1] + amount * (input[center + 1] - blurred_green));
    output[center + 2] = cudaimg::clamp_byte(input[center + 2] + amount * (input[center + 2] - blurred_blue));
    output[center + 3] = input[center + 3];
}

__global__ void sharpen_shared_kernel(const unsigned char* input, unsigned char* output, int width, int height, float amount) {
    constexpr int tile_width = cudaimg::kTileSize + 2;
    constexpr int tile_height = cudaimg::kTileSize + 2;
    __shared__ unsigned char tile[tile_width * tile_height * cudaimg::kChannels];

    const int block_x = blockIdx.x * cudaimg::kTileSize;
    const int block_y = blockIdx.y * cudaimg::kTileSize;
    const int thread_index = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    for (int index = thread_index; index < tile_width * tile_height; index += threads) {
        const int local_x = index % tile_width;
        const int local_y = index / tile_width;
        const int global_x = cudaimg::clamp_int(block_x + local_x - 1, 0, width - 1);
        const int global_y = cudaimg::clamp_int(block_y + local_y - 1, 0, height - 1);
        const int source = cudaimg::pixel_offset(global_x, global_y, width);
        const int target = (local_y * tile_width + local_x) * cudaimg::kChannels;
        tile[target + 0] = input[source + 0];
        tile[target + 1] = input[source + 1];
        tile[target + 2] = input[source + 2];
        tile[target + 3] = input[source + 3];
    }

    __syncthreads();

    const int x = block_x + threadIdx.x;
    const int y = block_y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    float blurred_red = 0.0f;
    float blurred_green = 0.0f;
    float blurred_blue = 0.0f;
    const int local_x = threadIdx.x + 1;
    const int local_y = threadIdx.y + 1;

    for (int kernel_y = -1; kernel_y <= 1; ++kernel_y) {
        for (int kernel_x = -1; kernel_x <= 1; ++kernel_x) {
            const int tile_index = ((local_y + kernel_y) * tile_width + (local_x + kernel_x)) * cudaimg::kChannels;
            const float weight = kKernelDevice[(kernel_y + 1) * 3 + (kernel_x + 1)];
            blurred_red += weight * tile[tile_index + 0];
            blurred_green += weight * tile[tile_index + 1];
            blurred_blue += weight * tile[tile_index + 2];
        }
    }

    blurred_red /= 16.0f;
    blurred_green /= 16.0f;
    blurred_blue /= 16.0f;

    const int output_index = cudaimg::pixel_offset(x, y, width);
    output[output_index + 0] = cudaimg::clamp_byte(tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 0] + amount * (tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 0] - blurred_red));
    output[output_index + 1] = cudaimg::clamp_byte(tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 1] + amount * (tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 1] - blurred_green));
    output[output_index + 2] = cudaimg::clamp_byte(tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 2] + amount * (tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 2] - blurred_blue));
    output[output_index + 3] = tile[((local_y)*tile_width + local_x) * cudaimg::kChannels + 3];
}

int run_sharpen_cpu_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const float amount = 0.8f + std::max(0.0f, strength) * 0.35f;
    const std::size_t bytes = cudaimg::image_bytes(width, height);
    std::vector<unsigned char> blurred(bytes);
    blur_once_cpu(input_rgba, blurred.data(), width, height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int index = cudaimg::pixel_offset(x, y, width);
            output_rgba[index + 0] = cudaimg::clamp_byte(input_rgba[index + 0] + amount * (input_rgba[index + 0] - blurred[index + 0]));
            output_rgba[index + 1] = cudaimg::clamp_byte(input_rgba[index + 1] + amount * (input_rgba[index + 1] - blurred[index + 1]));
            output_rgba[index + 2] = cudaimg::clamp_byte(input_rgba[index + 2] + amount * (input_rgba[index + 2] - blurred[index + 2]));
            output_rgba[index + 3] = input_rgba[index + 3];
        }
    }
    return 0;
}

int run_sharpen_cuda_naive_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const float amount = 0.8f + std::max(0.0f, strength) * 0.35f;
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [amount](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            sharpen_naive_kernel<<<grid, block>>>(buffers.input, buffers.output, launch_width, launch_height, amount);
            if (cudaGetLastError() != cudaSuccess) {
                return -1;
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                return -1;
            }
            return 0;
        });
}

int run_sharpen_cuda_optimized_impl(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    const float amount = 0.8f + std::max(0.0f, strength) * 0.35f;
    return cudaimg::run_cuda_operation(
        input_rgba,
        output_rgba,
        width,
        height,
        [amount](cudaimg::DeviceBuffers& buffers, int launch_width, int launch_height, std::size_t bytes) {
            const dim3 block = cudaimg::block_dim();
            const dim3 grid = cudaimg::grid_dim(launch_width, launch_height);
            sharpen_shared_kernel<<<grid, block>>>(buffers.input, buffers.output, launch_width, launch_height, amount);
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

extern "C" CUDA_IMAGE_API int sharpening_cpu(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sharpen_cpu_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int sharpening_cuda_naive(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sharpen_cuda_naive_impl(input_rgba, output_rgba, width, height, strength);
}

extern "C" CUDA_IMAGE_API int sharpening_cuda_optimized(const unsigned char* input_rgba, unsigned char* output_rgba, int width, int height, float strength) {
    return run_sharpen_cuda_optimized_impl(input_rgba, output_rgba, width, height, strength);
}
