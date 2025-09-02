```cpp
/* 
Aim: Modify the kernel so that only blocks with an odd index print their ID.

Thinking:
The requirement is to alter the kernel such that, during its execution, only the blocks
that have an odd index (i.e., blockIdx.x % 2 == 1) will print their block ID. 
In a typical CUDA program, a kernel may be launched with many blocks and each block
contains many threads. If we simply called printf from every thread, the output would
be highly redundant and noisy. Therefore, we restrict the printf call to a single
thread (usually threadIdx.x == 0) within each qualifying block.

The kernel will thus perform two checks:
1. `if (blockIdx.x % 2 == 1)` – ensures only odd-indexed blocks proceed.
2. `if (threadIdx.x == 0)` – ensures only the first thread in the block issues the
   printf, preventing duplicate prints from the same block.

The host code launches the kernel with a predefined number of blocks and threads.
After launching, we call `cudaDeviceSynchronize()` to wait for completion so that all
kernel printf output appears before the program exits. Finally, we reset the device
to clean up resources.

The program uses the standard `<cstdio>` header for `printf` and is compiled with
`nvcc`. It will compile and run on any CUDA-capable GPU. 
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints block ID only if the block index is odd
__global__ void printOddBlockID()
{
    // Check if this block has an odd index
    if (blockIdx.x % 2 == 1)
    {
        // Ensure only one thread per block does the printing
        if (threadIdx.x == 0)
        {
            printf("Block ID: %d\n", blockIdx.x);
        }
    }
}

int main()
{
    // Define number of blocks and threads per block
    const int numBlocks = 10;      // Example: 10 blocks
    const int threadsPerBlock = 32; // Example: 32 threads per block

    // Launch the kernel
    printOddBlockID<<<numBlocks, threadsPerBlock>>>();

    // Wait for the kernel to finish
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Clean up and reset the device
    cudaDeviceReset();

    return 0;
}
```