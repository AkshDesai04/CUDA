/*
Aim: Write a kernel where only thread 0 of a block prints a message (`if (threadIdx.x == 0) { ... }`). Launch with 128 threads.

Thinking:
- The task is to create a CUDA kernel that prints a message only from thread 0 of a block.
- We will use the built-in `printf` function available in device code. This requires compiling with `-arch=sm_XX` and linking against the CUDA runtime.
- To meet the "Launch with 128 threads" requirement, we will launch the kernel with a single block of 128 threads (`<<<1, 128>>>`).
- Inside the kernel, we check `if (threadIdx.x == 0)` and then call `printf`. This guarantees that only one thread prints, avoiding duplicate messages.
- For clarity, we also print the block index. Even though we use only one block, this demonstrates how to access `blockIdx.x`.
- The host code will launch the kernel, call `cudaDeviceSynchronize()` to ensure all output is flushed before the program exits.
- Minimal error checking is added for robustness, but the core requirement is fulfilled.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printFromThread0()
{
    if (threadIdx.x == 0)
    {
        printf("Hello from thread 0 of block %d\n", blockIdx.x);
    }
}

int main()
{
    // Launch the kernel with one block of 128 threads
    printFromThread0<<<1, 128>>>();
    
    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    
    return 0;
}
