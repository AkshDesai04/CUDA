/*
Aim: Also print the grid's dimension, `gridDim.x`, alongside the block index.

Thinking:
- We need a simple CUDA program that launches a kernel and prints the block index along with the grid dimension.
- Use `printf` inside the kernel to output `blockIdx.x` and `gridDim.x`.
- For clarity and to avoid flooding the console, choose a modest grid size (e.g., 4 blocks, 1 thread per block).
- Include error checking after kernel launch and a `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits.
- The program is written in standard C with CUDA extensions and should compile with `nvcc`.
- The output will show each block index and the constant grid dimension for all blocks.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_block_info(void)
{
    // Each block prints its index and the total number of blocks in the grid along the x-dimension
    printf("Block index: %d, gridDim.x: %d\n", blockIdx.x, gridDim.x);
}

int main(void)
{
    // Define grid and block dimensions
    dim3 gridDim(4, 1, 1);   // 4 blocks in x-dimension
    dim3 blockDim(1, 1, 1);  // 1 thread per block

    // Launch the kernel
    print_block_info<<<gridDim, blockDim>>>();

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Wait for the kernel to finish and flush all printf outputs
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Kernel execution completed successfully.\n");
    return EXIT_SUCCESS;
}
