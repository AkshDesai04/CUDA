/*
Aim: Launch a kernel with a 2D grid of blocks, for example, 4x4 blocks, each with a single thread. Each thread should print its 2D block index (`blockIdx.x`, `blockIdx.y`).

Thinking:
1. We need a CUDA kernel that runs one thread per block (blockDim = (1,1,1)).
2. Each thread prints its block index using device printf.
3. The grid dimensions should be 4x4 blocks: dim3 grid(4,4).
4. In main, set up the grid and block dimensions, launch the kernel, then synchronize.
5. Include basic error checking after the kernel launch to catch any launch failures.
6. Use `<cstdio>` for printf and `<cuda_runtime.h>` for CUDA runtime API.
7. Ensure the program compiles with nvcc and runs on a device that supports device-side printf.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Device-side printf to output the block indices
    printf("BlockIdx: (%d,%d)\n", (int)blockIdx.x, (int)blockIdx.y);
}

int main()
{
    // Define grid and block dimensions
    dim3 gridDim(4, 4);    // 4x4 blocks
    dim3 blockDim(1, 1);   // one thread per block

    // Launch kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Kernel execution completed successfully.\n");
    return EXIT_SUCCESS;
}
