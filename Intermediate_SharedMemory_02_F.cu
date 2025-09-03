```cuda
/*
Aim of the program (word to word from the prompt):
Explain in comments how using shared memory reduces the number of global memory reads.

Thinking process:
- The goal is to demonstrate that by loading data from global memory into shared memory once per thread block,
  each thread can reuse that data multiple times without fetching it again from global memory.
- This reduces the total number of global memory accesses, which is beneficial because global memory accesses
  are expensive in latency and bandwidth compared to shared memory accesses.
- A simple example is computing the sum of elements in an array using a reduction kernel.
  Without shared memory, each thread would read its element from global memory and then
  propagate sums across threads, potentially rereading the same data.
- With shared memory, each element is loaded once into shared memory by its corresponding thread.
  The reduction is performed entirely in shared memory, and the final result is written back to global memory once.
- In this program, the kernel will:
  1. Load each thread's element from global memory into a shared memory array.
  2. Perform a tree-based reduction within the block using only shared memory.
  3. Write the block's partial sum back to global memory.
  4. A host routine will then sum the partial sums to produce the final total.
- Comments inside the kernel explain how the shared memory usage reduces global reads and improves performance.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N (1 << 20)            // 1M elements
#define BLOCK_SIZE 256         // Threads per block

// Kernel that uses shared memory to sum an array.
// Each block computes a partial sum of a segment of the input array.
__global__ void reduce_sum_shared(const float *input, float *blockSums, int n)
{
    // Shared memory buffer for partial sums
    __shared__ float sdata[BLOCK_SIZE];

    // Global index of the thread
    unsigned int tid = threadIdx.x;
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load one element per thread from global memory into shared memory
    // This is the *only* global memory read per element for this block.
    float val = 0.0f;
    if (globalIdx < n) {
        val = input[globalIdx];
    }
    sdata[tid] = val;
    __syncthreads();

    /* Explanation of reduced global memory reads:
       - Each element of the input array is fetched from global memory exactly once.
       - All subsequent accesses to that element during the reduction are from shared memory.
       - Since shared memory resides on-chip with low latency and high bandwidth,
         this drastically cuts the number of expensive global memory reads.
       - The reduction algorithm below performs a binary tree summation
         using only shared memory accesses.
    */

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write the block's partial sum to global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}

int main()
{
    // Allocate host memory
    float *h_input = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f; // simple data

    // Allocate device memory
    float *d_input;
    float *d_blockSums;
    cudaMalloc((void**)&d_input, N * sizeof(float));

    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc((void**)&d_blockSums, numBlocks * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    reduce_sum_shared<<<numBlocks, BLOCK_SIZE>>>(d_input, d_blockSums, N);
    cudaDeviceSynchronize();

    // Copy partial sums back to host
    float *h_blockSums = (float*)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_blockSums, d_blockSums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final sum on host
    float total = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        total += h_blockSums[i];
    }

    printf("Sum = %f (should be %f)\n", total, (float)N);

    // Clean up
    free(h_input);
    free(h_blockSums);
    cudaFree(d_input);
    cudaFree(d_blockSums);

    return 0;
}
```