/*
Aim of the program: Modify the kernel so each thread prints `gridDim.x - 1 - blockIdx.x` (its block index from the end of the grid).

Thinking process:
- We want a simple CUDA program that demonstrates how to compute and print the index of a block relative to the end of the grid.
- The program will launch a kernel with a grid of multiple blocks; each block will have multiple threads.
- Inside the kernel, each thread will compute `gridDim.x - 1 - blockIdx.x`. Since this expression only depends on the block index, all threads within the same block will print the same value. This satisfies the requirement that each thread prints that value.
- The kernel will use `printf` (available in device code starting from CUDA 2.0) to output the computed value. We'll also print the thread and block indices for clarity.
- In the host code, we'll set up a small grid (e.g., 4 blocks of 4 threads each) to keep output manageable. After launching the kernel, we synchronize and exit.
- Error checking will be minimal for brevity, but a simple macro will be used to catch CUDA API errors.

The resulting file is a complete, compilable CUDA source (.cu) that prints the block index from the end for each thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

/* Kernel that prints gridDim.x - 1 - blockIdx.x for each thread */
__global__ void printBlockIndexFromEnd()
{
    int value = gridDim.x - 1 - blockIdx.x;
    printf("Thread %d in block %d prints %d\n", threadIdx.x, blockIdx.x, value);
}

int main(void)
{
    /* Define grid and block dimensions */
    const int threadsPerBlock = 4;
    const int blocksPerGrid   = 4;  // Adjust as desired

    /* Launch the kernel */
    printBlockIndexFromEnd<<<blocksPerGrid, threadsPerBlock>>>();

    /* Synchronize to ensure all printf output is flushed */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Reset the device before exiting */
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
