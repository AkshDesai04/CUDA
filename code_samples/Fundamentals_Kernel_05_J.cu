```cuda
/*
Aim: Explain in a comment why this formula for global ID works.

Thoughts:
In CUDA, the execution configuration is defined by the number of blocks (`gridDim`) and the number of threads per block (`blockDim`).  
`blockIdx.x` is the index of the current block along the X dimension, ranging from 0 to `gridDim.x - 1`.  
`threadIdx.x` is the index of the current thread within its block, ranging from 0 to `blockDim.x - 1`.

The global thread ID is computed as:
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

Why does this give a unique ID for every thread?

1. **Block offset**:  
   `blockIdx.x * blockDim.x` gives the starting global ID for the block.  
   For block 0, this is `0 * blockDim.x = 0`.  
   For block 1, it is `1 * blockDim.x = blockDim.x`.  
   For block 2, it is `2 * blockDim.x = 2 * blockDim.x`, and so on.  
   Thus each block’s threads are assigned a contiguous range of global IDs, and these ranges do not overlap because the multiplier is the block size.

2. **Thread offset within block**:  
   Adding `threadIdx.x` offsets the global ID within the block’s range.  
   For the first thread in the block (`threadIdx.x == 0`), the global ID remains the block offset.  
   For the last thread (`threadIdx.x == blockDim.x - 1`), the global ID becomes `blockIdx.x * blockDim.x + blockDim.x - 1`, which is exactly one less than the offset for the next block.

3. **Uniqueness**:  
   Because the block offsets are spaced by exactly `blockDim.x` and the thread offsets span `[0, blockDim.x - 1]`, every global ID is unique across all blocks and threads.  
   No two threads can produce the same `global_id` because that would require both a different block offset and a different thread offset that sum to the same value, which is impossible given the non‑overlapping ranges.

Thus, the formula guarantees that each thread obtains a distinct global ID that spans the entire grid, which is essential for correctly indexing global memory and ensuring proper parallel execution.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that prints the global ID of each thread
__global__ void print_global_id() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Global ID: %d\n", global_id);
}

int main(void) {
    const int threadsPerBlock = 256;
    const int numberOfBlocks = 10;

    // Launch kernel
    print_global_id<<<numberOfBlocks, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    return 0;
}
```