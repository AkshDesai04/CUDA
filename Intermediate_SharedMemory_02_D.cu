```cpp
/*
Aim of the program: 
Write the computation phase: after synchronizing, each thread computes its dot product using data exclusively from the shared memory tiles.

Thought process:
1. The goal is to demonstrate how to use shared memory to stage data for a dot product computation.
2. Each CUDA block will process a tile (chunk) of the input vectors.
3. Each thread loads one element from each vector into shared memory.
4. After a __syncthreads() barrier, all threads in the block can access the loaded tile.
5. Each thread then computes the product of its two corresponding elements (sA[tid] * sB[tid]) and accumulates it in a local variable.
6. A reduction within the block aggregates these partial products into a single block sum stored in global memory.
7. The host code orchestrates the allocation, data transfer, kernel launch, and final accumulation of block results.
8. The implementation is kept simple and self‑contained to compile as a .cu file, using only CUDA C++ APIs and no external dependencies.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 256  // Number of threads per block

// Kernel: Each block computes a partial dot product of a tile of the vectors.
__global__ void dotProductKernel(const float* __restrict__ A,
                                 const float* __restrict__ B,
                                 float* __restrict__ blockSums,
                                 int N)
{
    // Shared memory tiles for A and B
    __shared__ float sA[TILE_SIZE];
    __shared__ float sB[TILE_SIZE];

    // Thread index within the block
    int tid = threadIdx.x;

    // Global index for this thread
    int idx = blockIdx.x * TILE_SIZE + tid;

    // Load elements into shared memory if within bounds
    if (idx < N) {
        sA[tid] = A[idx];
        sB[tid] = B[idx];
    } else {
        // Pad with zeros if beyond vector length
        sA[tid] = 0.0f;
        sB[tid] = 0.0f;
    }

    // Synchronize so that all loads to shared memory are complete
    __syncthreads();

    // Each thread computes the product of its pair from the shared tile
    float localSum = sA[tid] * sB[tid];

    // Reduction within the block to accumulate local sums into blockSum
    // Use shared memory to hold partial sums during reduction
    __shared__ float partialSums[TILE_SIZE];
    partialSums[tid] = localSum;
    __syncthreads();

    // Classic reduction pattern (assumes TILE_SIZE is power of two)
    for (int stride = TILE_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSums[tid] += partialSums[tid + stride];
        }
        __syncthreads();
    }

    // The first thread writes the block's partial sum to global memory
    if (tid == 0) {
        blockSums[blockIdx.x] = partialSums[0];
    }
}

// Host function to perform dot product using the kernel
float dotProduct(const float* h_A, const float* h_B, int N)
{
    float *d_A, *d_B, *d_blockSums;
    int numBlocks = (N + TILE_SIZE - 1) / TILE_SIZE;
    size_t vecSize = N * sizeof(float);
    size_t blockSumsSize = numBlocks * sizeof(float);

    // Allocate device memory
    cudaMalloc((void**)&d_A, vecSize);
    cudaMalloc((void**)&d_B, vecSize);
    cudaMalloc((void**)&d_blockSums, blockSumsSize);

    // Copy vectors to device
    cudaMemcpy(d_A, h_A, vecSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, vecSize, cudaMemcpyHostToDevice);

    // Launch kernel
    dotProductKernel<<<numBlocks, TILE_SIZE>>>(d_A, d_B, d_blockSums, N);
    cudaDeviceSynchronize();

    // Copy block sums back to host
    float* h_blockSums = (float*)malloc(blockSumsSize);
    cudaMemcpy(h_blockSums, d_blockSums, blockSumsSize, cudaMemcpyDeviceToHost);

    // Final reduction on host
    float result = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        result += h_blockSums[i];
    }

    // Clean up
    free(h_blockSums);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_blockSums);

    return result;
}

int main()
{
    const int N = 1 << 20; // 1M elements
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));

    // Initialize vectors with some values
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;  // or any other pattern
        h_B[i] = 2.0f;
    }

    float result = dotProduct(h_A, h_B, N);
    printf("Dot product result: %f\n", result);

    free(h_A);
    free(h_B);

    return 0;
}
```