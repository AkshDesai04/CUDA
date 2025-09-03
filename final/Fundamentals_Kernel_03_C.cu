/*
Aim of the program:
Modify the kernel so each thread prints `blockDim.x - 1 - threadIdx.x` (its index from the end of the block).

Thinking:
1. The requirement is to change the kernel such that each thread computes its position relative to the last thread in its block.
2. The expression `blockDim.x - 1 - threadIdx.x` yields that value: for the first thread (threadIdx.x = 0) it prints `blockDim.x - 1`, for the last thread (threadIdx.x = blockDim.x - 1) it prints `0`.
3. The kernel will simply print that value using `printf`. Device-side printf requires `-rdc=true` when compiling, but most modern NVCC setups support it by default.
4. A minimal host code will launch a single block of a chosen size (e.g., 16 threads). After the kernel launch we call `cudaDeviceSynchronize()` to ensure all device printf output is flushed before the program exits.
5. The code includes necessary headers, error checking for the kernel launch, and a simple `int main()` that sets up the launch configuration, calls the kernel, synchronizes, and returns.
6. Since no device memory is needed, we avoid allocations for simplicity.
7. The final program is self‑contained and can be compiled with `nvcc -o index_from_end index_from_end.cu` and executed to observe the per‑thread index from the end of the block.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the index of each thread counted from the end of its block.
__global__ void printIndexFromEnd()
{
    // Compute index from the end: blockDim.x - 1 - threadIdx.x
    int indexFromEnd = blockDim.x - 1 - threadIdx.x;
    printf("Thread %d in block prints index from end: %d\n", threadIdx.x, indexFromEnd);
}

int main(void)
{
    // Define block size and number of blocks
    const int threadsPerBlock = 16;
    const int blocksPerGrid = 1;

    // Launch the kernel
    printIndexFromEnd<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Clean up and exit
    return 0;
}
