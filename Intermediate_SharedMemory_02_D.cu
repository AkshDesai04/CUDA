```cpp
/*
Write the computation phase: after synchronizing, each thread computes its dot product using data exclusively from the shared memory tiles.
The idea is to tile the vector dot product to exploit shared memory and reduce global memory traffic.
For each tile of size TILE_SIZE, each thread loads one element of vectors A and B into shared memory.
After synchronizing with __syncthreads(), every thread can safely read the entire tile from shared memory.
Each thread then multiplies its corresponding elements from sharedA and sharedB and accumulates a local partial sum.
A parallel reduction within the block aggregates these local sums into a block‑level result.
The kernel writes the block result to an output array; a separate kernel (or a host routine) can sum the block results to obtain the final dot product.
*/

// dot_product.cu

#include <cstdio>
#include <cuda_runtime.h>

#define TILE_SIZE 256  // Number of threads per block

// Kernel to compute partial dot products using shared memory tiles
__global__ void dotProductTile(const float *A, const float *B, float *partialSum, int N) {
    // Shared memory for a tile of A and B
    __shared__ float sharedA[TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * TILE_SIZE + tid;
    float localSum = 0.0f;

    // Process the vector in chunks of TILE_SIZE
    for (int i = gid; i < N; i += blockDim.x * gridDim.x) {
        // Load elements into shared memory
        sharedA[tid] = A[i];
        sharedB[tid] = B[i];
        __syncthreads();  // Ensure all loads complete before computation

        // ----- Computation phase -----
        // Each thread multiplies its tile element and accumulates into localSum
        localSum += sharedA[tid] * sharedB[tid];
        // --------------------------------

        __syncthreads();  // Optional: ensure all threads finished before next load
    }

    // Reduce localSum to a single value per block using a warp‑level reduction
    // (Assumes TILE_SIZE is a multiple of 32)
    for (int offset = TILE_SIZE/2; offset > 0; offset >>= 1) {
        float tmp = __shfl_down_sync(0xffffffff, localSum, offset);
        localSum += tmp;
    }

    // Thread 0 writes the block's partial sum
    if (tid == 0) {
        partialSum[blockIdx.x] = localSum;
    }
}

// Host function to launch kernel and compute final dot product
float dotProduct(const float *h_A, const float *h_B, int N) {
    float *d_A, *d_B, *d_partial;
    int numBlocks = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate device memory
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_partial, numBlocks * sizeof(float));

    // Copy input vectors to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    dotProductTile<<<numBlocks, TILE_SIZE>>>(d_A, d_B, d_partial, N);
    cudaDeviceSynchronize();

    // Copy partial sums back to host
    float *h_partial = new float[numBlocks];
    cudaMemcpy(h_partial, d_partial, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final reduction on host
    float result = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        result += h_partial[i];
    }

    // Clean up
    delete[] h_partial;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial);

    return result;
}

// Example usage
int main() {
    const int N = 1 << 20;  // 1M elements
    float *h_A = new float[N];
    float *h_B = new float[N];

    // Initialize vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    float result = dotProduct(h_A, h_B, N);
    printf("Dot product = %f\n", result);

    delete[] h_A;
    delete[] h_B;
    return 0;
}
```