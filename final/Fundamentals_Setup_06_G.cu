```cpp
/*
Aim of the program: Explain in a comment the difference between block dimensions and grid dimensions.

Thinking:
In CUDA, the execution configuration of a kernel is specified by a grid of thread blocks.
- The **grid dimension** (gridDim) specifies how many blocks are launched in each dimension (x, y, z).
  Each block in the grid is identified by its block index (blockIdx) and has its own independent execution.
- The **block dimension** (blockDim) specifies how many threads are launched within each block in each dimension.
  Threads within the same block share resources such as shared memory and can synchronize with each other.

These two concepts are independent:
  * The grid dimension determines the total number of blocks, thus affecting the total number of
    thread blocks that can be executed in parallel across the GPU.
  * The block dimension determines how many threads are in each block, affecting intra-block parallelism
    and the amount of resources each block consumes.

This program demonstrates the distinction by launching a kernel with a specific grid and block size,
and then printing the block indices and block dimensions from the host side.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that records block indices and block dimensions for each block.
// Only the first thread in each block writes the data to avoid race conditions.
__global__ void recordBlockInfo(int *blockIdxX, int *blockIdxY, int *blockIdxZ,
                                int *blockDimX, int *blockDimY, int *blockDimZ)
{
    if (threadIdx.x == 0) { // Only one thread per block writes the info
        // Compute a linear index for the block
        int idx = blockIdx.x
                + blockIdx.y * gridDim.x
                + blockIdx.z * gridDim.x * gridDim.y;

        blockIdxX[idx] = blockIdx.x;
        blockIdxY[idx] = blockIdx.y;
        blockIdxZ[idx] = blockIdx.z;

        blockDimX[idx] = blockDim.x;
        blockDimY[idx] = blockDim.y;
        blockDimZ[idx] = blockDim.z;
    }
}

int main(void)
{
    // Define grid and block dimensions
    dim3 gridDim(2, 3, 1);   // 2 blocks in x, 3 blocks in y, 1 block in z
    dim3 blockDim(4, 4, 2);  // 4 threads in x, 4 threads in y, 2 threads in z

    int totalBlocks = gridDim.x * gridDim.y * gridDim.z;

    // Allocate host arrays to hold block information
    int *h_blockIdxX = (int*)malloc(totalBlocks * sizeof(int));
    int *h_blockIdxY = (int*)malloc(totalBlocks * sizeof(int));
    int *h_blockIdxZ = (int*)malloc(totalBlocks * sizeof(int));
    int *h_blockDimX = (int*)malloc(totalBlocks * sizeof(int));
    int *h_blockDimY = (int*)malloc(totalBlocks * sizeof(int));
    int *h_blockDimZ = (int*)malloc(totalBlocks * sizeof(int));

    // Allocate device arrays
    int *d_blockIdxX, *d_blockIdxY, *d_blockIdxZ;
    int *d_blockDimX, *d_blockDimY, *d_blockDimZ;
    cudaMalloc((void**)&d_blockIdxX, totalBlocks * sizeof(int));
    cudaMalloc((void**)&d_blockIdxY, totalBlocks * sizeof(int));
    cudaMalloc((void**)&d_blockIdxZ, totalBlocks * sizeof(int));
    cudaMalloc((void**)&d_blockDimX, totalBlocks * sizeof(int));
    cudaMalloc((void**)&d_blockDimY, totalBlocks * sizeof(int));
    cudaMalloc((void**)&d_blockDimZ, totalBlocks * sizeof(int));

    // Launch kernel
    recordBlockInfo<<<gridDim, blockDim>>>(d_blockIdxX, d_blockIdxY, d_blockIdxZ,
                                           d_blockDimX, d_blockDimY, d_blockDimZ);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_blockIdxX, d_blockIdxX, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockIdxY, d_blockIdxY, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockIdxZ, d_blockIdxZ, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockDimX, d_blockDimX, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockDimY, d_blockDimY, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockDimZ, d_blockDimZ, totalBlocks * sizeof(int), cudaMemcpyDeviceToHost);

    // Print block indices and dimensions
    printf("Block information (total blocks = %d):\n", totalBlocks);
    for (int i = 0; i < totalBlocks; ++i) {
        printf("Block %d: idx=(%d,%d,%d), dim=(%d,%d,%d)\n",
               i,
               h_blockIdxX[i], h_blockIdxY[i], h_blockIdxZ[i],
               h_blockDimX[i], h_blockDimY[i], h_blockDimZ[i]);
    }

    // Clean up
    free(h_blockIdxX); free(h_blockIdxY); free(h_blockIdxZ);
    free(h_blockDimX); free(h_blockDimY); free(h_blockDimZ);
    cudaFree(d_blockIdxX); cudaFree(d_blockIdxY); cudaFree(d_blockIdxZ);
    cudaFree(d_blockDimX); cudaFree(d_blockDimY); cudaFree(d_blockDimZ);

    return 0;
}
```