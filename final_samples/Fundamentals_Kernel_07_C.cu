/*
Also print the block's 2D dimensions (`blockDim.x`, `blockDim.y`).

Thought process:
- The goal is to create a simple CUDA C program that prints the block's 2D dimensions.
- I will write a kernel that uses `printf` from device code, which is supported on compute capability 2.0 and higher.
- The kernel will print the thread index and the block's dimensions (`blockDim.x` and `blockDim.y`).
- In `main`, I will set a grid of 2x2 blocks, each with a 8x8 2D block.
- After launching the kernel, I will call `cudaDeviceSynchronize()` to ensure all prints finish before the program exits.
- The code will be self‑contained and compile to a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockDimKernel()
{
    // Print the block dimensions for each thread in the block
    printf("Thread (%d,%d) in block (%d,%d) reports blockDim: (%d,%d)\n",
           threadIdx.x, threadIdx.y,
           blockIdx.x, blockIdx.y,
           blockDim.x, blockDim.y);
}

int main(void)
{
    // Define 2D block and grid dimensions
    dim3 blockDim(8, 8);  // 8x8 threads per block
    dim3 gridDim(2, 2);   // 2x2 blocks in the grid

    // Launch the kernel
    printBlockDimKernel<<<gridDim, blockDim>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
