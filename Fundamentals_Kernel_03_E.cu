```cuda
/*
Aim of the program: 
Modify the kernel so that only threads in the first half of the block (i.e., `threadIdx.x < blockDim.x / 2`) print their ID.

Thinking:
- The original program likely prints the ID of every thread in the block.
- The requirement is to restrict printing to only the first half of the block.
- Inside the kernel, we check the condition `threadIdx.x < blockDim.x / 2` before calling printf.
- `blockDim.x` is the number of threads per block, so `blockDim.x / 2` gives the half-point.
- Since integer division truncates, this works for both even and odd numbers of threads:
    - If blockDim.x is odd, `blockDim.x / 2` is the floor of half, which correctly gives the lower half.
- We keep the rest of the kernel unchanged aside from the conditional.
- The host code launches the kernel with a chosen number of threads per block (e.g., 16) and one block.
- After kernel launch we synchronize and check for errors.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints thread ID only for threads in the first half of the block
__global__ void printThreadID()
{
    // Only print if the thread index is in the first half of the block
    if (threadIdx.x < blockDim.x / 2)
    {
        printf("Thread ID: %d\n", threadIdx.x);
    }
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 16;   // Number of threads in each block
    const int blocksPerGrid   = 1;    // Number of blocks

    // Launch the kernel
    printThreadID<<<blocksPerGrid, threadsPerBlock>>>();

    // Synchronize to ensure all prints are done before program exits
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Check for any errors that may have occurred during kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```