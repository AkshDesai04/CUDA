/*
Launch with a 3D grid of (2, 2, 2) blocks, each with one thread. Have each thread print its `blockIdx.x`, `blockIdx.y`, and `blockIdx.z`.

Thinking:
- Use a simple CUDA kernel that calls `printf` to output the block indices.
- The grid is defined as `dim3 grid(2, 2, 2);` and the block size as `dim3 block(1, 1, 1);`.
- Since there is only one thread per block, each thread will execute the kernel once and print its block index.
- After launching the kernel, we need to call `cudaDeviceSynchronize()` to ensure that all device prints are flushed before the host program exits.
- Include basic error checking for the kernel launch and the device synchronization to catch any launch failures.
- The code is written for CUDA 10+ where device-side `printf` is supported.
- The output will be printed to the console, one line per thread, indicating the block indices.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple macro for error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that prints blockIdx.x, blockIdx.y, blockIdx.z
__global__ void printBlockIdx()
{
    printf("Thread blockIdx: (%d, %d, %d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main(void)
{
    // Define a 3D grid of (2,2,2) blocks
    dim3 gridDim(2, 2, 2);
    // Each block has only one thread
    dim3 blockDim(1, 1, 1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Check for any launch errors
    CUDA_CHECK(cudaGetLastError());

    // Wait for device to finish and flush printf output
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Kernel execution completed.\n");
    return 0;
}
