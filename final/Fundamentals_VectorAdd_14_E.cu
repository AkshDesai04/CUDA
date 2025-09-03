/*
Verify the `double` precision results on the CPU. Note the potential for tiny floating point discrepancies.

The goal of this program is to demonstrate that a simple numerical computation performed on the GPU using CUDA
can be verified against a CPU implementation, even when using `double` precision floating‑point arithmetic.
Because of the way floating‑point numbers are rounded and accumulated in parallel, the GPU and CPU results may
differ by a very small amount (on the order of 1e‑12 or less).  This program:
  1. Computes the sum of 1/(i+1)² for i = 0 … N-1 on the GPU.
  2. Computes the same sum on the CPU.
  3. Prints both results and their absolute difference.
  4. Reports that the difference is within a reasonable tolerance.

To avoid issues with atomic operations on older GPUs, the GPU sum is computed using a two‑step reduction:
   - Each block computes a partial sum and writes it to a block‑sums array.
   - A second kernel reduces the block‑sums array into a single global sum.

The code includes basic CUDA error checking and is self‑contained.  It can be compiled with
  nvcc -arch=sm_20 -o double_sum double_sum.cu
and run on any GPU that supports double precision.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define N 10000000          // Number of terms to sum
#define BLOCK_SIZE 256      // Threads per block

// Kernel to compute 1/(i+1)^2 and accumulate partial sums per block
__global__ void computePartialSums(double *blockSums, int N)
{
    __shared__ double shared[BLOCK_SIZE];
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;

    if (globalIdx < N) {
        double x = (double)(globalIdx + 1);
        val = 1.0 / (x * x);
    }
    shared[threadIdx.x] = val;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Write block's partial sum to global memory
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = shared[0];
    }
}

// Kernel to reduce block sums into a single global sum
__global__ void reduceBlockSums(double *blockSums, double *globalSum, int numBlocks)
{
    __shared__ double shared[BLOCK_SIZE];
    int idx = threadIdx.x;
    double val = 0.0;
    if (idx < numBlocks) {
        val = blockSums[idx];
    }
    shared[idx] = val;
    __syncthreads();

    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (idx < stride) {
            shared[idx] += shared[idx + stride];
        }
        __syncthreads();
    }

    // First thread writes final sum
    if (idx == 0) {
        *globalSum = shared[0];
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    // Determine grid size
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate memory for block sums on device
    double *d_blockSums;
    CUDA_CHECK(cudaMalloc((void**)&d_blockSums, gridSize * sizeof(double)));

    // Device variable for final sum
    double *d_globalSum;
    CUDA_CHECK(cudaMalloc((void**)&d_globalSum, sizeof(double)));

    // Launch kernel to compute partial sums
    computePartialSums<<<gridSize, BLOCK_SIZE>>>(d_blockSums, N);
    CUDA_CHECK(cudaGetLastError());

    // Reduce block sums into global sum
    // Use a single block with BLOCK_SIZE threads
    int reduceThreads = (gridSize < BLOCK_SIZE) ? gridSize : BLOCK_SIZE;
    reduceBlockSums<<<1, reduceThreads>>>(d_blockSums, d_globalSum, gridSize);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    double gpuSum = 0.0;
    CUDA_CHECK(cudaMemcpy(&gpuSum, d_globalSum, sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_globalSum));

    // Compute sum on CPU
    double cpuSum = 0.0;
    for (int i = 0; i < N; ++i) {
        double x = (double)(i + 1);
        cpuSum += 1.0 / (x * x);
    }

    // Print results
    printf("CPU sum:   %.15f\n", cpuSum);
    printf("GPU sum:   %.15f\n", gpuSum);
    printf("Difference: %.15e\n", fabs(cpuSum - gpuSum));

    // Check if difference is within a small tolerance
    double tolerance = 1e-12;
    if (fabs(cpuSum - gpuSum) <= tolerance) {
        printf("Result is within tolerance of %.1e.\n", tolerance);
    } else {
        printf("Result differs by more than tolerance of %.1e.\n", tolerance);
    }

    return 0;
}
