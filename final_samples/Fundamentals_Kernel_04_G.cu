/*
Aim: Launch with a 3D grid of (2, 2, 2) blocks, each with one thread. Have each thread print its `blockIdx.x`, `blockIdx.y`, and `blockIdx.z`.

Thinking:
- The requirement specifies a 3‑D grid where each dimension has size 2, giving a total of 2*2*2 = 8 blocks.
- Each block must contain exactly one thread, so the block dimension is (1,1,1).
- The kernel will use `printf` to output the block indices. CUDA allows `printf` in device code, but the output appears asynchronously; thus we must call `cudaDeviceSynchronize()` after launching the kernel to ensure all prints are flushed before the program exits.
- We add simple error checking after kernel launch and device sync to catch any CUDA runtime errors.
- The program uses `nvcc`‑compatible headers (`<cuda_runtime.h>` and `<stdio.h>`).
- The main function sets the grid and block dimensions, launches the kernel, synchronizes, and then exits.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printBlockIndices()
{
    // Each thread prints its block index components.
    printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n",
           blockIdx.x, blockIdx.y, blockIdx.z);
}

int main()
{
    // Define a 3D grid of (2, 2, 2) blocks.
    dim3 grid(2, 2, 2);
    // Each block has only one thread.
    dim3 block(1, 1, 1);

    // Launch the kernel.
    printBlockIndices<<<grid, block>>>();

    // Check for launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Synchronize to ensure all printf output is flushed.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA sync failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
