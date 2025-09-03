/*
Launch the printing kernel with one block of 10 threads. Observe the output.

The goal of this small CUDA program is to demonstrate how to launch a kernel with a
specific grid and block configuration and how device-side printf works.  We define a
kernel that simply prints the thread index and block index of each thread.  In the
main function we set the grid dimensions to one block and the block dimensions
to ten threads.  After launching the kernel we call cudaDeviceSynchronize to make
sure all output is flushed before the host program exits.

This example helps to understand:
1. How to specify grid/block launch parameters.
2. How many threads actually run (10 in this case).
3. The order of printf output is not guaranteed because threads run concurrently.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the thread and block indices
__global__ void printKernel(void)
{
    // Each thread prints its own threadIdx.x and blockIdx.x
    printf("Thread %d in block %d\n", threadIdx.x, blockIdx.x);
}

int main(void)
{
    // Launch parameters: 1 block, 10 threads per block
    int threadsPerBlock = 10;
    int blocksPerGrid    = 1;

    // Launch the kernel
    printKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for any launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing any results or exiting
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optionally reset the device
    cudaDeviceReset();

    return 0;
}
