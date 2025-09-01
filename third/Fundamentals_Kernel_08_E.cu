/*
Aim of the program: Combine with 2D blocks. Launch a 2x2 grid of 4x4 blocks. Have each thread calculate and print its global 2D index: `int global_x = blockIdx.x * blockDim.x + threadIdx.x;` and `int global_y = ...`.

Thinking process:
- The task requires a CUDA kernel that uses a 2D grid and 2D blocks.
- We choose a grid dimension of (2, 2) and a block dimension of (4, 4).
- Each thread must compute its global coordinates:
    global_x = blockIdx.x * blockDim.x + threadIdx.x
    global_y = blockIdx.y * blockDim.y + threadIdx.y
- To observe the result, we use device-side printf to print each thread's block-relative and global coordinates.
- After kernel launch we call cudaDeviceSynchronize() to ensure all printf outputs are flushed before the program exits.
- Standard headers and basic error checking are included for completeness.
- The kernel is launched from the host main function, and the program is self-contained in a single .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printGlobalIndices()
{
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Print block-relative and global coordinates
    printf("Thread (%d, %d) -> Global (%d, %d)\n",
           threadIdx.x, threadIdx.y,
           global_x, global_y);
}

int main(void)
{
    // Define 2D grid and block dimensions
    dim3 gridDim(2, 2);   // 2x2 grid
    dim3 blockDim(4, 4);  // 4x4 blocks

    // Launch kernel
    printGlobalIndices<<<gridDim, blockDim>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
