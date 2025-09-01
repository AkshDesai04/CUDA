/*
Aim: Explain in a comment the difference between block dimensions and grid dimensions.

This program demonstrates the concept of block and grid dimensions in CUDA programming.
In CUDA, a kernel launch is organized into a two‑level hierarchy: a grid of blocks,
and each block contains a set of threads. The grid dimensions (`gridDim`) specify
how many blocks are launched in each dimension (x, y, z). The block dimensions
(`blockDim`) specify how many threads are inside each block in each dimension.

Key differences:
1. **Scope**:
   - `gridDim` defines the total number of blocks in the grid.
   - `blockDim` defines the total number of threads in each block.

2. **Indexing**:
   - Each block has a unique index `blockIdx` (x, y, z). Together with `gridDim`,
     this gives the block’s position in the entire grid.
   - Each thread within a block has a unique index `threadIdx` (x, y, z). Together
     with `blockIdx` and `blockDim`, this gives the thread’s unique position
     across the entire grid (`blockIdx * blockDim + threadIdx`).

3. **Limits**:
   - The maximum grid dimensions depend on the GPU architecture (e.g., up to
     2^31-1 blocks in one dimension for many GPUs).
   - The maximum block dimensions are limited by the device (commonly 1024 threads
     per block, but also constrained by shared memory and register usage).

4. **Usage**:
   - Use `gridDim` when you need to partition work across many blocks (e.g., to
     process a large array).
   - Use `blockDim` to pack enough threads into each block to fully utilize the
     SMs and to allow efficient intra‑block communication (shared memory, sync).

By understanding and properly choosing both grid and block dimensions, one can
balance parallelism, memory access patterns, and resource usage for optimal
performance on CUDA-enabled GPUs.
*/

// Simple CUDA program illustrating block and grid dimensions

#include <stdio.h>

__global__ void printIndices()
{
    // Compute a unique global thread index
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Print block and thread indices
    printf("Global Thread ID: %d | BlockIdx.x: %d | ThreadIdx.x: %d | "
           "BlockDim.x: %d | GridDim.x: %d\n",
           global_thread_id, blockIdx.x, threadIdx.x,
           blockDim.x, gridDim.x);
}

int main()
{
    // Define block and grid dimensions
    int threadsPerBlock = 8;   // Block dimension (threads per block)
    int blocksPerGrid   = 4;   // Grid dimension (blocks per grid)

    // Launch kernel
    printIndices<<<blocksPerGrid, threadsPerBlock>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}
