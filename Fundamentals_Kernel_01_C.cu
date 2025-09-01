/*
Use `dim3` variables to define a grid of (4,1,1) and a block of (1,1,1) and launch the kernel.

In this program we create a very simple CUDA kernel that prints the block and thread indices.
The aim of the program is to demonstrate how to set up grid and block dimensions using
`dim3` variables. We define a grid with 4 blocks along the x-dimension and a block
with a single thread. After launching the kernel, we synchronize the device to ensure
that all printf output is flushed before the program exits. No memory allocations
are performed; the kernel operates purely on launch parameters. The code is fully
self‑contained and can be compiled with `nvcc` to produce an executable that
demonstrates the specified launch configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    // Print block and thread indices
    printf("BlockIdx.x = %d, ThreadIdx.x = %d\n", blockIdx.x, threadIdx.x);
}

int main()
{
    // Define grid and block dimensions using dim3
    dim3 grid(4, 1, 1);   // 4 blocks in x-dimension
    dim3 block(1, 1, 1);  // 1 thread per block

    // Launch the kernel
    printKernel<<<grid, block>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA launch error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
