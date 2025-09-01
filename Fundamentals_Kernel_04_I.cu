/*
Aim: Launch with 16 blocks. Have each thread print its block ID multiplied by 10.

Thinking:
- The requirement specifies launching with exactly 16 blocks, so we set the grid dimension to 16.
- Each thread should output its block ID times 10. Since the output is per thread, we can decide how many threads per block. The simplest approach is to use one thread per block (threadsPerBlock = 1) to avoid redundant prints and keep the output concise. If more threads were used per block, each thread would print the same value (blockIdx.x * 10) multiple times, which is unnecessary given the wording.
- The kernel will call `printf` with the computed value. Device-side `printf` is available on modern CUDA-capable GPUs, but it requires a printf-capable device and should be compiled with the appropriate architecture flags.
- After launching the kernel, we must call `cudaDeviceSynchronize()` to ensure all device prints are flushed before the program exits.
- Error checking after the kernel launch will help catch any launch failures.
- The host code includes the standard includes `<stdio.h>` and `<cuda_runtime.h>`. We also set the device explicitly to 0 for consistency.
- For simplicity, we use `cudaGetLastError()` to capture any launch errors and print them to the host console.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockID()
{
    // Each thread prints its block ID multiplied by 10
    printf("%d\n", blockIdx.x * 10);
}

int main(void)
{
    const int numBlocks = 16;
    const int threadsPerBlock = 1; // One thread per block for clear output

    // Launch kernel
    printBlockID<<<numBlocks, threadsPerBlock>>>();

    // Synchronize to ensure all printf outputs are completed
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
