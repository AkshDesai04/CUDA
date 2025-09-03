/*
Explain in a comment why this formula for global ID works.

In a CUDA kernel, threads are organized into blocks. Each block has a
certain number of threads (blockDim.x). The index of a thread within its
block is given by threadIdx.x. Blocks themselves are indexed by
blockIdx.x. The global thread ID is the unique identifier of a thread
across the entire grid. 

The formula:
    globalIdx = blockIdx.x * blockDim.x + threadIdx.x

works because:
1. blockIdx.x * blockDim.x calculates how many threads are present in all
   blocks that come before the current block. This is the offset at the
   start of the current block in the global indexing scheme.
2. Adding threadIdx.x gives the thread's position within the current block.
   Since threadIdx.x ranges from 0 to blockDim.x-1, the sum is a unique
   integer from 0 to (number of threads in grid)-1.

Thus the formula provides a contiguous, zero‑based global index that
maps one‑to‑one to each thread in the grid.

The following program demonstrates this by computing the global index
for each thread and storing it in an array that is then printed on the
host.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void computeGlobalIDs(int *ids, int n) {
    // Compute the global thread index
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < n) {
        ids[globalIdx] = globalIdx;
    }
}

int main(void) {
    const int N = 16; // Total number of threads to launch
    int *h_ids = (int*)malloc(N * sizeof(int));
    int *d_ids;
    cudaMalloc((void**)&d_ids, N * sizeof(int));

    // Launch configuration
    const int threadsPerBlock = 4;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Execute kernel
    computeGlobalIDs<<<blocks, threadsPerBlock>>>(d_ids, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_ids, d_ids, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the global IDs
    printf("Global IDs for %d threads:\n", N);
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_ids[i]);
    }
    printf("\n");

    // Cleanup
    cudaFree(d_ids);
    free(h_ids);

    return 0;
}
