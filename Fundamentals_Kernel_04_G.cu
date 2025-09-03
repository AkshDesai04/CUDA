/*
Aim: Launch with a 3D grid of (2, 2, 2) blocks, each with one thread. Have each thread print its `blockIdx.x`, `blockIdx.y`, and `blockIdx.z`.

Thinking:
1. We need a CUDA kernel that runs one thread per block, so `blockDim = dim3(1)`.
2. The grid is 3D with dimensions (2, 2, 2) → `gridDim = dim3(2, 2, 2)`.
3. Inside the kernel, use `printf` (device-side) to output the block indices.  
   This requires `#include <stdio.h>` and compiling with `nvcc` which automatically enables device printf.
4. After launching the kernel, call `cudaDeviceSynchronize()` to ensure all prints are flushed to the console before the program exits.
5. Add simple error checking after the kernel launch and synchronization for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each block has only one thread
    printf("BlockIdx: (%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    // Define a 3D grid with dimensions (2, 2, 2)
    dim3 gridDim(2, 2, 2);
    // Each block contains exactly one thread
    dim3 blockDim(1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Clean up and exit
    return EXIT_SUCCESS;
}
