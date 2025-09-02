```cuda
/* 
This reduction pattern has bank conflicts in the first few iterations. A conflict-free version has the stride `s` decrease from `blockDim.x / 2` but the access is `s_data[i] += s_data[i+s]`. Compare the performance. 

Thinking:
The goal is to demonstrate how shared‑memory bank conflicts can degrade performance in a parallel reduction.
A classic reduction kernel (naiveReduction) uses shared memory without padding; when the stride `s` becomes a multiple of the number of shared‑memory banks (32 on most GPUs), threads in the same warp will contend for the same bank, causing serialization. 

To eliminate this conflict we use a padded shared memory array (conflictFreeReduction). Padding the array by one element (`__shared__ float s_data[BLOCK_SIZE+1];`) shifts the addresses that each thread accesses by one, ensuring that the memory accesses are distributed across banks even when the stride is a multiple of 32.  

Both kernels perform the same logical operation: each block reduces its chunk of the input array to a single partial sum, and the host then reduces these partial sums. 
We compare execution times using CUDA events and verify that the results are identical. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define BLOCK_SIZE 256   // Must be power of two

// Naive reduction kernel (likely to have bank conflicts)
__global__ void naiveReduction(const float *input, float *output, size_t n) {
    extern __shared__ float s_data[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread to reduce the number of blocks
    float sum = 0.0f;
    if (idx < n)       sum = input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    s_data[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_data[tid] += s_data[tid + s];
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = s_data[0];
}

// Conflict‑free reduction kernel using padded shared memory
__global__ void conflictFreeReduction(const float *input, float *output, size_t n) {
    __shared__ float s_data[BLOCK_SIZE + 1];   // Pad by 1 to avoid bank conflicts
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load two elements per thread
    float sum = 0.0f;
    if (idx < n)       sum = input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    s_data[tid + 1] = sum;   // Offset by 1
    __syncthreads();

    // Reduction in shared memory (padded)
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            s_data[tid + 1] += s_data[tid + 1 + s];
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = s_data[1];
}

// Host-side reduction of partial sums
float hostReduce(const float *partial, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += partial[i];
    return sum;
}

int main(int argc, char *argv[]) {
    size_t N = 1 << 24;   // ~16M elements (~64MB)
    if (argc > 1) N = strtoul(argv[1], nullptr, 10);

    // Allocate host memory
    float *h_input = (float *)malloc(N * sizeof(float));
    float *h_partial = (float *)malloc(((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2)) * sizeof(float));

    // Initialize input with random data
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i)
        h_input[i] = (float)(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_input, *d_partial;
    CHECK_CUDA(cudaMalloc((void **)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_partial, h_partial ? h_partial : NULL));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Determine grid size
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + block.x * 2 - 1) / (block.x * 2));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // -------- Naive reduction ----------
    CHECK_CUDA(cudaEventRecord(start));
    naiveReduction<<<grid, block, BLOCK_SIZE * sizeof(float)>>>(d_input, d_partial, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msNaive;
    CHECK_CUDA(cudaEventElapsedTime(&msNaive, start, stop));

    // Copy partial sums back and reduce on host
    size_t partialCount = grid.x;
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, partialCount * sizeof(float), cudaMemcpyDeviceToHost));
    float sumNaive = hostReduce(h_partial, partialCount);

    // -------- Conflict‑free reduction ----------
    CHECK_CUDA(cudaEventRecord(start));
    conflictFreeReduction<<<grid, block, (BLOCK_SIZE + 1) * sizeof(float)>>>(d_input, d_partial, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msConflictFree;
    CHECK_CUDA(cudaEventElapsedTime(&msConflictFree, start, stop));

    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, partialCount * sizeof(float), cudaMemcpyDeviceToHost));
    float sumConflictFree = hostReduce(h_partial, partialCount);

    // Verify results
    if (fabs(sumNaive - sumConflictFree) > 1e-5f) {
        fprintf(stderr, "Results differ! Naive=%f, ConflictFree=%f\n", sumNaive, sumConflictFree);
    } else {
        printf("Results match: %f\n", sumNaive);
    }

    // Print timings
    printf("Naive reduction time:     %6.3f ms\n", msNaive);
    printf("Conflict‑free time:       %6.3f ms\n", msConflictFree);

    // Clean up
    free(h_input);
    free(h_partial);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_partial));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```