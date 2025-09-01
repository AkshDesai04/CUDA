/*
Aim of the program: Also print the block's 2D dimensions (blockDim.x, blockDim.y).

Thinking process:
- The goal is to demonstrate how to access and print the block dimensions in a CUDA kernel.
- The kernel will launch with a 2D block configuration so that blockDim.x and blockDim.y are non‑trivial.
- To avoid cluttering the output with duplicate messages from every thread, only the first thread of each block (threadIdx.x == 0 && threadIdx.y == 0) will perform the printf.
- The host code will set up a grid of 2×2 blocks, each block having dimensions 16×8 threads.
- We use device printf, which requires the device architecture to support it (compute capability ≥ 2.0). The code will compile with standard nvcc options.
- After kernel execution we synchronize and then exit.
- This simple program serves as an educational example to illustrate blockDim.x and blockDim.y.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockDim()
{
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
    }
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(16, 8);   // 16 threads in X, 8 threads in Y
    dim3 gridDim(2, 2);     // 2 blocks in X, 2 blocks in Y

    // Launch kernel
    printBlockDim<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA synchronize failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
