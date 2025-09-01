/*
Aim: Also print the grid's dimension, `gridDim.x`, alongside the block index.

Thinking:
- The goal is to demonstrate that a CUDA kernel can access both the block index (blockIdx.x) and the grid dimension (gridDim.x).
- We'll launch a kernel with a small number of blocks so that the output is easy to read. Each block will print its own index together with the total number of blocks in the grid.
- To avoid cluttering the output, only the first thread of each block (threadIdx.x == 0) will perform the printf. This ensures each block prints once.
- The kernel will use `printf` which is supported on GPUs that have compute capability 2.0 or higher.
- After launching the kernel, the host will synchronize and check for errors.
- For simplicity, we will use a block dimension of 1 thread per block and a grid dimension of 4 blocks. This can be easily modified if needed.
- The program includes basic error checking for the kernel launch and for device synchronization.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints block index and grid dimension
__global__ void printBlockInfo()
{
    // Only thread 0 of each block prints
    if (threadIdx.x == 0) {
        printf("Block index: %d, Grid dimension (x): %d\n", blockIdx.x, gridDim.x);
    }
}

int main()
{
    // Define grid and block dimensions
    const int blocks = 4;
    const int threadsPerBlock = 1;

    // Launch kernel
    printBlockInfo<<<blocks, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Clean up and exit
    return 0;
}
