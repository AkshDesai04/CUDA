/*
Compare the performance of the boundary-checked kernel on a perfect-multiple size (e.g. 512x512) vs. the original kernel without checks. Is there overhead?

Thinking:
- Use a simple array operation (e.g., add 1) to benchmark.
- Allocate 512x512 float array.
- Implement two kernels:
  1. kernel_no_check: assumes perfect multiple; no bounds check.
  2. kernel_with_check: includes if (x < width && y < height) guard.
- Launch both kernels with same grid/block dims: 16x16 blocks, 32x32 grid.
- Use CUDA events to time each kernel execution.
- Use cudaMemset to zero or initialize memory.
- After kernel runs, copy back result to host if needed (not necessary for timing).
- Print execution times and ratio.
- Ensure correct cleanup.

Edge considerations:
- Use cudaEventRecord and cudaEventSynchronize.
- Use cudaMemcpy to transfer data; but we can allocate device memory only.
- Use a simple operation so kernel code is trivial.
- Ensure device synchronization before measuring times.

We'll output results to console.

*/

#include <cstdio>
#include <cuda_runtime.h>

#define WIDTH 512
#define HEIGHT 512

// Kernel without boundary checks
__global__ void kernel_no_check(float *data, int width, int height)
{
    // Compute 2D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Linear index
    int idx = y * width + x;

    // Perform a simple operation
    data[idx] += 1.0f;
}

// Kernel with boundary checks
__global__ void kernel_with_check(float *data, int width, int height)
{
    // Compute 2D indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check
    if (x < width && y < height)
    {
        int idx = y * width + x;
        data[idx] += 1.0f;
    }
}

int main()
{
    size_t size = WIDTH * HEIGHT * sizeof(float);
    float *d_data = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_data, size);
    if (err != cudaSuccess)
    {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Initialize data to zero
    err = cudaMemset(d_data, 0, size);
    if (err != cudaSuccess)
    {
        printf("cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return -1;
    }

    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x,
                  (HEIGHT + blockSize.y - 1) / blockSize.y);

    // Events for timing
    cudaEvent_t startNoCheck, stopNoCheck;
    cudaEvent_t startCheck, stopCheck;
    cudaEventCreate(&startNoCheck);
    cudaEventCreate(&stopNoCheck);
    cudaEventCreate(&startCheck);
    cudaEventCreate(&stopCheck);

    // Launch kernel without boundary check
    cudaEventRecord(startNoCheck, 0);
    kernel_no_check<<<gridSize, blockSize>>>(d_data, WIDTH, HEIGHT);
    cudaEventRecord(stopNoCheck, 0);

    // Wait for completion
    cudaEventSynchronize(stopNoCheck);

    // Launch kernel with boundary check
    cudaEventRecord(startCheck, 0);
    kernel_with_check<<<gridSize, blockSize>>>(d_data, WIDTH, HEIGHT);
    cudaEventRecord(stopCheck, 0);

    // Wait for completion
    cudaEventSynchronize(stopCheck);

    // Compute elapsed times
    float msNoCheck = 0.0f, msCheck = 0.0f;
    cudaEventElapsedTime(&msNoCheck, startNoCheck, stopNoCheck);
    cudaEventElapsedTime(&msCheck, startCheck, stopCheck);

    printf("Kernel without boundary check: %f ms\n", msNoCheck);
    printf("Kernel with boundary check:    %f ms\n", msCheck);
    printf("Overhead ratio (check/no check): %f\n", msCheck / msNoCheck);

    // Clean up
    cudaEventDestroy(startNoCheck);
    cudaEventDestroy(stopNoCheck);
    cudaEventDestroy(startCheck);
    cudaEventDestroy(stopCheck);
    cudaFree(d_data);

    return 0;
}
