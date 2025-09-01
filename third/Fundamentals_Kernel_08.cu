/*
Launch a kernel with a 2D grid of blocks, for example, 4x4 blocks, each with a single thread. Each thread should print its 2D block index (blockIdx.x, blockIdx.y).

Thinking:
- The goal is to demonstrate a simple CUDA kernel that prints the block indices of each thread.
- Since each block will contain only one thread, the kernel will be invoked with a block dimension of (1,1).
- The grid will be a 4x4 arrangement of blocks, giving a total of 16 threads.
- Each thread uses the built-in `printf` function to output its `blockIdx.x` and `blockIdx.y` values.
- After launching the kernel, we must call `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits.
- A minimal, self‑contained CUDA program will include the necessary headers, the kernel definition, and the `main` function that sets up and launches the kernel.
- The code is written in standard CUDA C++ syntax and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each thread prints its block index
    printf("BlockIdx: (%d, %d)\n", blockIdx.x, blockIdx.y);
}

int main()
{
    // Define a 4x4 grid of blocks, each block has 1 thread
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1);

    // Launch the kernel
    printBlockIdx<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Optional: Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
