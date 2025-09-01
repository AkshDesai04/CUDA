/*
Change the launch configuration to a non-square 2D block, like 16x4 threads, and print the 2D indices.
[thinking]
The task requires modifying a CUDA kernel launch configuration to use a non-square 2D block size of 16x4 threads. 
We want to demonstrate how to print out both block and thread indices in 2D form. 
The simplest approach is to use device-side printf (available in recent CUDA versions) within the kernel.
We will define a kernel that prints the block and thread indices for each thread.
On the host side, we will launch this kernel with a blockDim of (16,4) and a single grid block.
After launching, we will synchronize and reset the device to clean up.
The code includes basic error checking for CUDA API calls. 
We will use standard headers <cstdio> and <cuda_runtime.h>.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

__global__ void printIndices()
{
    // Print block and thread indices in 2D
    printf("Block (%d,%d) Thread (%d,%d)\n",
           blockIdx.x, blockIdx.y,
           threadIdx.x, threadIdx.y);
}

int main()
{
    // Define block and grid dimensions
    dim3 blockDim(16, 4, 1);   // 16x4 threads
    dim3 gridDim(1, 1, 1);     // Single block for simplicity

    // Launch kernel
    printIndices<<<gridDim, blockDim>>>();

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Synchronize to ensure all printf output is flushed
    CHECK_CUDA(cudaDeviceSynchronize());

    // Reset device (optional)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
