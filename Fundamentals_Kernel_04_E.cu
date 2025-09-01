/*
Aim of the program:
Launch the kernel with a 2D grid of (4, 4) blocks (each with one thread) and have each thread print only its blockIdx.x.

Thinking:
- The kernel will simply use printf to output the value of blockIdx.x for each thread.
- We set up a 2‑D grid with dimensions (4,4) and each block will contain exactly one thread.
- Since each block has only one thread, threadIdx.x will always be 0, but we are interested in blockIdx.x.
- The kernel will be launched and then we call cudaDeviceSynchronize() to ensure all device output is flushed before the program exits.
- The code includes minimal error checking for clarity; it compiles with nvcc and runs on any CUDA device that supports device-side printf (compute capability ≥ 2.0).
- The expected output is 16 lines, each showing the blockIdx.x value (0,1,2,3 repeated four times each).
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIdxX()
{
    // Each thread prints only its blockIdx.x
    printf("BlockIdx.x: %d\n", blockIdx.x);
}

int main()
{
    // Define a 2D grid of (4, 4) blocks, each block has one thread
    dim3 gridDim(4, 4);
    dim3 blockDim(1, 1, 1);

    // Launch the kernel
    printBlockIdxX<<<gridDim, blockDim>>>();

    // Wait for the GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}
