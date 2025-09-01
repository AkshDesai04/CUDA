/*
 * Aim: Launch a kernel with a 2D block of threads, for example, 8x8 threads. Each thread should print its 2D thread index (`threadIdx.x`, `threadIdx.y`).
 *
 * Thinking:
 * - Use a CUDA kernel that prints threadIdx.x and threadIdx.y using device printf.
 * - Launch a single block of size dim3(8,8) and no grid dimension >1.
 * - Use cudaDeviceSynchronize to ensure all prints are flushed before exiting.
 * - Check for launch errors.
 * - The kernel uses compute capability 2.0 or higher for device printf support.
 */

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printThreadIndices()
{
    // Each thread prints its 2D index
    printf("Thread (%d, %d)\n", threadIdx.x, threadIdx.y);
}

int main()
{
    // Define block size: 8x8 threads
    dim3 blockSize(8, 8);

    // Launch one block
    printThreadIndices<<<1, blockSize>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
