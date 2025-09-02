/*
Modify the kernel so each thread prints `blockDim.x - 1 - threadIdx.x` (its index from the end of the block).

Thoughts:
- The goal is to compute, for each thread, its position relative to the end of its block. This is simply `blockDim.x - 1 - threadIdx.x`.
- We'll use `printf` from device code to print the block index, thread index, and the computed value.
- A simple error checking routine will be added after kernel launch to capture any launch failures.
- The kernel will be launched with a small grid and block size (e.g., 2 blocks of 8 threads) for easy visualization.
- After the kernel completes, `cudaDeviceSynchronize` ensures all prints have been flushed.
- The program is written in plain C for CUDA (`.cu` file) and uses the standard `stdio.h` for output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printIndexFromEndKernel()
{
    int idxFromEnd = blockDim.x - 1 - threadIdx.x;
    printf("Block %d, Thread %d, Index from end: %d\n",
           blockIdx.x, threadIdx.x, idxFromEnd);
}

int checkCudaError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main(void)
{
    const int threadsPerBlock = 8;
    const int blocksPerGrid = 2;

    // Launch the kernel
    printIndexFromEndKernel<<<blocksPerGrid, threadsPerBlock>>>();
    if (checkCudaError("Kernel launch failed")) return 1;

    // Wait for the kernel to finish and ensure all device prints are flushed
    cudaDeviceSynchronize();
    if (checkCudaError("cudaDeviceSynchronize failed")) return 1;

    return 0;
}
