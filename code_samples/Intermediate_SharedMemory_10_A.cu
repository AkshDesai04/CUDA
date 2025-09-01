/*
The kernel should load a tile from `input` into shared memory `s_tile` such that `s_tile[threadIdx.y][threadIdx.x] = input[...]`. This read is coalesced.

Thinking:
- We want a generic CUDA program that demonstrates loading a tile of an image or matrix into shared memory in a coalesced manner.
- Use a 2D grid of 2D blocks. Each block will load a TILE_DIM x TILE_DIM tile from global memory.
- Each thread in the block corresponds to one element in the tile, so threadIdx.x and threadIdx.y map directly to the shared memory indices.
- Compute the global coordinates (gx, gy) for each thread as blockIdx.x * TILE_DIM + threadIdx.x and blockIdx.y * TILE_DIM + threadIdx.y.
- Perform a bounds check to ensure we don't read beyond the input dimensions.
- Since threads in the same warp read contiguous memory addresses (difference of 1 in the x direction), the read is fully coalesced on modern GPUs.
- Allocate shared memory as a 2D array: __shared__ float s_tile[TILE_DIM][TILE_DIM];. Note that this array is placed in shared memory and is used by all threads in the block.
- After loading, we can optionally synchronize threads (but not required for the load itself).
- For completeness, the host code will allocate device memory, copy data, launch the kernel, and copy back a result buffer (even though the result buffer is not used here, it demonstrates full round-trip).
- We use constexpr for TILE_DIM and other constants.
- The kernel is called TileLoadKernel, and we set blockDim to TILE_DIM x TILE_DIM, gridDim based on input dimensions.
- The main function sets up a simple example: a 1024x1024 float array initialized to some values, launches the kernel, and then cleans up.
- Compile with: nvcc -o tile_load tile_load.cu
*/

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

constexpr int TILE_DIM = 32; // Tile dimension; must match blockDim.x and blockDim.y

// Kernel that loads a TILE_DIM x TILE_DIM tile from global memory into shared memory
__global__ void TileLoadKernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int width, int height)
{
    // Allocate shared memory tile
    __shared__ float s_tile[TILE_DIM][TILE_DIM];

    // Global coordinates of the element this thread will load
    int gx = blockIdx.x * TILE_DIM + threadIdx.x;
    int gy = blockIdx.y * TILE_DIM + threadIdx.y;

    // Bounds check
    if (gx < width && gy < height)
    {
        // Compute linear index in input array
        int input_idx = gy * width + gx;
        // Load into shared memory tile
        s_tile[threadIdx.y][threadIdx.x] = input[input_idx];
    }
    else
    {
        // If out of bounds, we can set shared memory to zero or leave unchanged
        s_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize to make sure all loads are finished before using shared memory
    __syncthreads();

    // For demonstration, copy the shared memory tile back to global memory
    // This step is optional and only to illustrate that data is in shared memory.
    if (gx < width && gy < height)
    {
        int output_idx = gy * width + gx;
        output[output_idx] = s_tile[threadIdx.y][threadIdx.x];
    }
}

int main()
{
    // Define image dimensions
    const int width = 1024;
    const int height = 1024;
    const int size = width * height;
    const size_t bytes = size * sizeof(float);

    // Host data
    std::vector<float> h_input(size);
    std::vector<float> h_output(size, 0.0f);

    // Initialize input with some data
    for (int i = 0; i < size; ++i)
    {
        h_input[i] = static_cast<float>(i);
    }

    // Device pointers
    float* d_input = nullptr;
    float* d_output = nullptr;

    // Allocate device memory
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    TileLoadKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost);

    // Simple check: print a few values
    std::cout << "Sample output values:" << std::endl;
    for (int y = 0; y < 5; ++y)
    {
        for (int x = 0; x < 5; ++x)
        {
            std::cout << h_output[y * width + x] << " ";
        }
        std::cout << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
