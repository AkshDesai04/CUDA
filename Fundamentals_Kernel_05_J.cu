/*
Aim: Explain in a comment why this formula for global ID works.

Thinking:
The goal of this small CUDA program is to illustrate how to compute a unique global thread
identifier (ID) in a 1‑D grid of thread blocks.  In CUDA each thread belongs to a block,
and each block belongs to a grid.  The built‑in variables blockIdx.x, blockDim.x, and
threadIdx.x give, respectively, the block’s index along the grid, the number of threads
in each block, and the thread’s index within its block.  When we want a single integer
that uniquely identifies a thread across the entire grid, we combine these pieces of
information into the expression:

    global_id = blockIdx.x * blockDim.x + threadIdx.x

The multiplication term, blockIdx.x * blockDim.x, gives the number of threads that
precede the current block in the grid.  Each block contains blockDim.x threads, so
if the current block is, say, block 3 (blockIdx.x == 3) and each block has 128
threads (blockDim.x == 128), then there are 3 * 128 = 384 threads before block 3.
Adding threadIdx.x, which is the offset of the current thread within its own block,
yields the overall position of the thread in the flattened grid.  This simple linear
formula works because CUDA blocks are laid out in a row‑major order along each
dimension of the grid, and we are using only the first dimension (x).  The result
is a one‑to‑one mapping between the pair (blockIdx.x, threadIdx.x) and the
unique global_id.  This mapping is used throughout CUDA for indexing arrays,
implementing parallel reductions, and many other patterns.

The program below demonstrates this calculation and prints the global ID of each
thread.  It also prints the values of blockIdx.x, blockDim.x, and threadIdx.x
to show how they contribute to the global ID.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_global_ids()
{
    // Compute the unique global ID for each thread
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Print the global ID along with the components for clarity
    printf("BlockIdx.x = %3d, ThreadIdx.x = %3d, GlobalID = %4d\n",
           blockIdx.x, threadIdx.x, global_id);
}

int main(void)
{
    // Define grid and block dimensions
    const int threads_per_block = 8;
    const int number_of_blocks   = 4;

    // Launch kernel
    print_global_ids<<<number_of_blocks, threads_per_block>>>();

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    return 0;
}
