```c
/*
Aim of the program:
For the final 32 elements (one warp), `__syncthreads` is not needed between iterations. Use warp shuffle intrinsics (`__shfl_down_sync`) for a highly optimized final stage of the reduction.

Thinking:
- The goal is to implement a parallel reduction kernel that sums an array of floats.
- The reduction is performed in two phases:
  1. A shared‑memory reduction where each block reduces its portion of the input array into a single partial sum. During this phase, threads cooperate and must be synchronized with `__syncthreads()` after each step that uses shared memory.
  2. A final warp‑level reduction that operates on the last 32 (one warp) elements of the shared array. For this phase, we can use warp shuffle intrinsics (`__shfl_down_sync`) to avoid a costly `__syncthreads()` barrier, because all threads in the warp already have the data in registers.
- The kernel must handle arbitrary input size `n`. Each thread will load multiple elements using a grid‑stride loop.
- We will use a block size that is a multiple of the warp size (e.g., 512). Each block writes its partial sum to an output array in global memory.
- A second kernel (or a host function) will then reduce the partial sums to a single result. For simplicity we can launch a second kernel that does the same reduction but with a small number of blocks (e.g., 1 block).
- The host code allocates memory, initializes input, copies data to device, launches the reduction kernel(s), copies back the result, and verifies correctness.
- The code is written in CUDA C++ and is fully self‑contained, compiling with `nvcc`.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512   // Must be a multiple of warp size (32)
#define WARP_SIZE 32

// Device kernel: reduction of a float array to a single sum per block
__global__ void reduceSum(const float *in, float *out, int n)
{
    // Shared memory buffer for partial sums
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Each thread loads multiple elements into shared memory
    float sum = 0.0f;
    // Grid stride loop to cover all elements
    for (int i = idx; i < n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }

    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    // First perform pairwise reductions up to the warp size
    for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Warp-level reduction using shuffle intrinsics
    // No need for __syncthreads() between iterations
    if (tid < WARP_SIZE) {
        // Load the warp's portion of shared memory into registers
        float val = sdata[tid];
        // Reduce within the warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            // Use mask to avoid accessing out of bounds in case blockDim.x < 32
            int mask = 0xffffffff;
            // For blocks smaller than warp size, mask out the inactive lanes
            if (blockDim.x < WARP_SIZE) {
                mask = (1 << blockDim.x) - 1;
            }
            val += __shfl_down_sync(mask, val, offset);
        }
        // Thread 0 writes the block's partial sum to global memory
        if (tid == 0) out[blockIdx.x] = val;
    }
}

// Host function to perform full reduction using two kernels
float reduceArray(const float *h_in, int n)
{
    float *d_in, *d_partial, *d_out;
    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    size_t size = n * sizeof(float);
    size_t partialSize = numBlocks * sizeof(float);

    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_partial, partialSize);
    cudaMalloc((void**)&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // First kernel: reduce each block to a partial sum
    reduceSum<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
        d_in, d_partial, n);
    cudaDeviceSynchronize();

    // If more than one block produced partial sums, reduce them again
    // Launch a second kernel with one block
    if (numBlocks > 1) {
        reduceSum<<<1, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_partial, d_out, numBlocks);
        cudaDeviceSynchronize();
    } else {
        // Only one block, copy result directly
        cudaMemcpy(d_out, d_partial, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);

    return h_out;
}

// Simple test
int main()
{
    const int N = 1 << 20; // 1M elements
    float *h_in = (float*)malloc(N * sizeof(float));

    // Initialize input data
    for (int i = 0; i < N; ++i) h_in[i] = 1.0f; // sum should be N

    float result = reduceArray(h_in, N);
    printf("Sum of %d elements = %f\n", N, result);

    free(h_in);
    return 0;
}
```