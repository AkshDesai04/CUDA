```c
/*
Launch the printing kernel with 10 blocks of one thread. Observe the output.

Thought process:
1. The goal is to create a CUDA kernel that prints a message from each thread.
2. We want to launch the kernel with 10 blocks and 1 thread per block.
3. CUDA device code can use printf, which prints to stdout.
4. The kernel will compute its block and thread indices and output them.
5. In the host code, we launch the kernel with the desired configuration,
   synchronize, and check for errors.
6. No additional libraries are needed beyond <cstdio> and <cuda_runtime.h>.
7. The program is a minimal, self-contained .cu file that can be compiled
   with nvcc and run to see the output from each block/thread.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printKernel()
{
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;
    printf("Hello from block %d, thread %d\n", blockId, threadId);
}

int main()
{
    // Launch kernel with 10 blocks, 1 thread per block
    printKernel<<<10, 1>>>();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for the kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```