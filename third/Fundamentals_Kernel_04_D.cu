```cpp
/*
Aim of the program:
Modify the kernel so that only blocks with an odd index print their ID.

Thinking:
To accomplish this, we create a simple CUDA program that launches a grid of blocks.
Each block will attempt to print its block index (blockIdx.x). However, we only want
blocks with an odd index to perform the print. We can achieve this by adding a
conditional statement inside the kernel: if (blockIdx.x % 2 == 1) { printf(...); }
We use device-side printf (requires device capability >= 2.0) to output the block ID.
After launching the kernel, we synchronize the device to ensure all prints are
flushed before the program exits. This minimal example demonstrates the conditional
execution of kernel code based on block indices.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printOddBlockIds()
{
    // Only blocks with odd blockIdx.x print their ID
    if (blockIdx.x % 2 == 1)
    {
        printf("Block ID: %d\n", blockIdx.x);
    }
}

int main()
{
    const int numBlocks = 8;   // Example grid size
    const int threadsPerBlock = 1;

    // Launch kernel
    printOddBlockIds<<<numBlocks, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```