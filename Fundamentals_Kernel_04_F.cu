/*
Aim of the program:
Launch with a 1D grid of 8 blocks, but give each block 4 threads. Have only thread 0 of each block print the `blockIdx.x`.

Thought process:
1. We want a simple CUDA program that demonstrates how to launch a kernel with a specific grid and block configuration.
2. The grid dimension is 1D with 8 blocks, so the gridDim.x will be 8.
3. Each block should contain 4 threads, so threadDim.x will be 4.
4. Inside the kernel we need to identify thread 0 of each block. This is done by checking `threadIdx.x == 0`.
5. Only that thread should perform a printf to display the block index (`blockIdx.x`). This avoids cluttering the output with multiple prints per block.
6. Because `printf` is used on the device, we must synchronize the device after kernel launch to ensure all output is flushed before the host program exits. This is done with `cudaDeviceSynchronize()`.
7. The host code is minimal: include <stdio.h>, define the kernel, launch it with the chosen grid/block dimensions, synchronize, and return 0.
8. The code is written in standard CUDA C, suitable for compiling with `nvcc`.
*/

#include <stdio.h>

__global__ void printBlockIdx(void)
{
    // Only the first thread of each block prints the block index
    if (threadIdx.x == 0)
    {
        printf("Block %d\n", blockIdx.x);
    }
}

int main(void)
{
    // Launch the kernel with 8 blocks and 4 threads per block
    printBlockIdx<<<8, 4>>>();

    // Ensure all device printf output is flushed before the program exits
    cudaDeviceSynchronize();

    return 0;
}
