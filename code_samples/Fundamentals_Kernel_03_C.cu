/*
Aim: Modify the kernel so each thread prints `blockDim.x - 1 - threadIdx.x` (its index from the end of the block).

Thinking:
1. The user wants a CUDA program where each thread in a block prints its reverse index, i.e., how far it is from the last thread in the block. This is computed as `blockDim.x - 1 - threadIdx.x`.
2. We'll implement a simple kernel that uses device-side printf to output the thread index and its reverse index.
3. In the host code, we’ll launch the kernel with a single block of a chosen size (e.g., 8 threads) for demonstration.
4. After launching, we’ll call `cudaDeviceSynchronize()` to ensure all prints are flushed before the program exits.
5. The code is self‑contained and can be compiled with `nvcc -arch=sm_52 example.cu -o example`.
6. The comment at the top includes both the aim (exact wording from the prompt) and the reasoning steps taken to design the program.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Kernel that prints each thread's index from the end of the block
__global__ void printReverseIndices()
{
    // Compute reverse index: blockDim.x - 1 - threadIdx.x
    int reverseIdx = blockDim.x - 1 - threadIdx.x;
    printf("Thread %d prints reverse index %d\n", threadIdx.x, reverseIdx);
}

int main()
{
    // Define block and grid dimensions
    const int threadsPerBlock = 8;
    const int blocksPerGrid = 1;

    // Launch the kernel
    printReverseIndices<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
