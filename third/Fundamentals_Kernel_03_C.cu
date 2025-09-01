/*
Modify the kernel so each thread prints `blockDim.x - 1 - threadIdx.x` (its index from the end of the block).

The goal of this program is to demonstrate how to calculate the reverse index of a thread within its block and print that value from device code. The kernel will launch a few blocks with a fixed number of threads, and each thread will output the computed reverse index. After kernel launch we synchronize and check for any errors to ensure the output is produced correctly.

Key points considered:
1. Use `blockDim.x` and `threadIdx.x` to compute the reverse index.
2. Use `printf` in device code – requires the compute capability to support device printf.
3. Host code will set up grid and block dimensions, launch the kernel, and synchronize.
4. Basic error handling is included for clarity.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the reverse index of each thread within its block
__global__ void reverseIndexKernel()
{
    // Calculate reverse index: blockDim.x - 1 - threadIdx.x
    int reverseIdx = blockDim.x - 1 - threadIdx.x;
    printf("Block %d, Thread %d -> reverse index %d\n",
           blockIdx.x, threadIdx.x, reverseIdx);
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 4;
    const int blocksPerGrid = 2;

    // Launch kernel
    reverseIndexKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Reset device (optional)
    cudaDeviceReset();

    return 0;
}
