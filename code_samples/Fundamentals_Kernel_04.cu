/*
Write a kernel that is launched with a 1D grid of 8 blocks, where each block has only one thread. Each thread should print its own block index, which is accessible via blockIdx.x.

My thinking:
- The requirement is to launch a 1D grid of 8 blocks, each with a single thread, so <<<8, 1>>>.
- Each thread prints blockIdx.x. In CUDA, device printf is available, but we need to call cudaDeviceSynchronize to ensure all prints complete before the program exits.
- Use a simple __global__ kernel that calls printf with blockIdx.x.
- In main, launch the kernel and sync.
- The code should compile as a .cu file and run on a device that supports printf.
- No additional features needed; just core logic.
*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void printBlockIdx()
{
    // Each thread in this kernel is responsible for printing its block index.
    printf("Block index: %d\n", blockIdx.x);
}

int main()
{
    // Launch the kernel with 8 blocks and 1 thread per block.
    printBlockIdx<<<8, 1>>>();

    // Wait for the kernel to finish so that all printf outputs are flushed.
    cudaDeviceSynchronize();

    return 0;
}
