```cuda
/*
Aim of the program:
Print the values of `blockDim.x` and `gridDim.x` from within the kernel.

Thinking:
The goal is to demonstrate how device-side variables such as blockDim.x and gridDim.x can be accessed
and printed from within a CUDA kernel. The simplest way to do this is to launch a kernel with
known dimensions (for example, grid of 2 blocks and each block containing 1 thread) and use
`printf` inside the kernel to output the values. Since `printf` from the device is asynchronous
with respect to the host, we need to call `cudaDeviceSynchronize()` after the kernel launch to
ensure the output is flushed before the program exits. The host code is straightforward:
include `<cstdio>`, define the kernel, launch it with chosen grid and block dimensions, synchronize,
and exit. Basic error checking is added to make sure the kernel launch and synchronization succeed.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printDimsKernel()
{
    // Each thread prints the block and grid dimensions along the x axis
    printf("Thread %d of block %d: blockDim.x = %d, gridDim.x = %d\n",
           threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);
}

int main()
{
    // Define grid and block dimensions
    dim3 blockDim(1);   // One thread per block
    dim3 gridDim(2);    // Two blocks in the grid

    // Launch the kernel
    printDimsKernel<<<gridDim, blockDim>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Synchronize to ensure all device printf outputs are flushed
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Successful completion
    return EXIT_SUCCESS;
}
```