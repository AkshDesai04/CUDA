/*
Aim: Use `atomicAdd` to safely have multiple threads increment a counter member within the single shared struct instance.
Thinking: 
- We create a struct SharedCounter with a single integer counter.
- Each block has one instance of this struct in shared memory.
- Thread 0 initializes the counter to zero.
- All threads in the block call atomicAdd on the address of s.counter to increment it.
- After synchronization, thread 0 writes the final counter value to a global array.
- The host launches multiple blocks, each with multiple threads, and then prints the counts from each block.
- This demonstrates atomicAdd usage on a shared memory struct member.
*/

#include <stdio.h>
#include <cuda_runtime.h>

typedef struct {
    int counter;
} SharedCounter;

__global__ void counterKernel(int threadsPerBlock, int *blockCounts)
{
    __shared__ SharedCounter s;

    // Initialize counter once per block
    if (threadIdx.x == 0) {
        s.counter = 0;
    }
    __syncthreads();

    // Each thread increments the counter atomically
    atomicAdd(&(s.counter), 1);

    __syncthreads();

    // Store the result for this block
    if (threadIdx.x == 0) {
        blockCounts[blockIdx.x] = s.counter;
    }
}

int main(void)
{
    const int numBlocks = 10;
    const int threadsPerBlock = 32;

    int *d_blockCounts = NULL;
    int *h_blockCounts = NULL;

    size_t countsSize = numBlocks * sizeof(int);

    // Allocate device memory for block counts
    cudaError_t err = cudaMalloc((void**)&d_blockCounts, countsSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Allocate host memory for results
    h_blockCounts = (int*)malloc(countsSize);
    if (!h_blockCounts) {
        fprintf(stderr, "Host malloc failed\n");
        cudaFree(d_blockCounts);
        return 1;
    }

    // Launch kernel
    counterKernel<<<numBlocks, threadsPerBlock>>>(threadsPerBlock, d_blockCounts);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        free(h_blockCounts);
        cudaFree(d_blockCounts);
        return 1;
    }

    // Copy results back to host
    err = cudaMemcpy(h_blockCounts, d_blockCounts, countsSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        free(h_blockCounts);
        cudaFree(d_blockCounts);
        return 1;
    }

    // Print results
    printf("Block counts:\n");
    for (int i = 0; i < numBlocks; ++i) {
        printf("  Block %2d: %d\n", i, h_blockCounts[i]);
    }

    // Clean up
    free(h_blockCounts);
    cudaFree(d_blockCounts);

    return 0;
}