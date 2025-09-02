```cuda
/*
Use `atomicMax` to implement a single-kernel max reduction and compare its performance to the two-pass method.

Thinking:
- The goal is to showcase how atomic operations can be used for a global reduction in a single kernel launch, and to benchmark it against a more traditional two-pass reduction where the first pass computes block-level maxima and the second pass reduces those to a final result.
- We'll work with an integer array (int) because CUDA provides `atomicMax` for integral types. Using floats would require reinterpret casting which adds unnecessary complexity for the demonstration.
- The array size will be configurable, but for performance testing we'll use a large array (e.g., 1<<24 elements). This ensures enough work for the GPU to hide latency.
- For the single-kernel reduction:
  * Each thread loads its element, performs a local max within a warp/block using shared memory.
  * The first thread of each block writes its block maximum to a global variable `d_globalMax` via `atomicMax`.
  * After the kernel completes, we copy `d_globalMax` back to the host.
- For the two-pass reduction:
  * First kernel writes block maxima into an array `d_blockMax`.
  * Second kernel reduces `d_blockMax` (size = number of blocks) to a single maximum using the same shared memory reduction pattern.
- Timing: We'll use CUDA events (`cudaEvent_t`) for accurate GPU timing of each kernel. CPU-side allocation, data transfer, and error checking are negligible compared to kernel execution for large arrays.
- We also verify that both methods produce the same maximum value.
- The code is self-contained, uses only CUDA runtime API, and is compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define ARRAY_SIZE (1 << 24)  // 16 million elements
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel: Single-pass reduction using atomicMax
__global__ void singleKernelMax(const int* d_in, int* d_globalMax, size_t N) {
    __shared__ int sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localMax = INT_MIN;

    // Load elements and compute local max
    while (idx < N) {
        int val = d_in[idx];
        if (val > localMax) localMax = val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = localMax;
    __syncthreads();

    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the block max to global memory via atomicMax
    if (tid == 0) {
        atomicMax(d_globalMax, sdata[0]);
    }
}

// Kernel: First pass of two-pass reduction (block-wise maxima)
__global__ void firstPassMax(const int* d_in, int* d_blockMax, size_t N) {
    __shared__ int sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localMax = INT_MIN;

    // Load elements and compute local max
    while (idx < N) {
        int val = d_in[idx];
        if (val > localMax) localMax = val;
        idx += blockDim.x * gridDim.x;
    }

    sdata[tid] = localMax;
    __syncthreads();

    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block maximum to global array
    if (tid == 0) {
        d_blockMax[blockIdx.x] = sdata[0];
    }
}

// Kernel: Second pass of two-pass reduction (reduces block maxima)
__global__ void secondPassMax(const int* d_blockMax, int* d_finalMax, size_t blockMaxSize) {
    __shared__ int sdata[THREADS_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x;

    int localMax = INT_MIN;

    // Each thread loads one block max (if available)
    if (idx < blockMaxSize) {
        localMax = d_blockMax[idx];
    }

    sdata[tid] = localMax;
    __syncthreads();

    // Reduction within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid])
                sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }

    // The first thread writes the final maximum
    if (tid == 0) {
        d_finalMax[0] = sdata[0];
    }
}

// Helper to generate random integer array
void generateRandomIntArray(int* h_array, size_t N) {
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < N; ++i) {
        h_array[i] = rand() % 1000000;  // random ints in [0, 999999]
    }
}

int main(void) {
    size_t N = ARRAY_SIZE;
    size_t bytes = N * sizeof(int);

    // Allocate host memory
    int* h_in = (int*)malloc(bytes);
    if (!h_in) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Generate random data
    generateRandomIntArray(h_in, N);

    // Allocate device memory
    int* d_in = nullptr;
    int* d_globalMax = nullptr;
    int* d_blockMax = nullptr;
    int* d_finalMax = nullptr;

    CHECK_CUDA(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_globalMax, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_finalMax, sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Initialize global max to INT_MIN
    int initVal = INT_MIN;
    CHECK_CUDA(cudaMemcpy(d_globalMax, &initVal, sizeof(int), cudaMemcpyHostToDevice));

    // Determine grid size
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 65535) blocks = 65535; // maximum grid dimension in 1D

    // Allocate array for block maxima for two-pass method
    CHECK_CUDA(cudaMalloc((void**)&d_blockMax, blocks * sizeof(int)));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Single-kernel atomicMax reduction
    CHECK_CUDA(cudaEventRecord(start));
    singleKernelMax<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_globalMax, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msSingle = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msSingle, start, stop));

    // Copy result
    int hostSingleMax;
    CHECK_CUDA(cudaMemcpy(&hostSingleMax, d_globalMax, sizeof(int), cudaMemcpyDeviceToHost));

    // Two-pass reduction
    // First pass
    CHECK_CUDA(cudaEventRecord(start));
    firstPassMax<<<blocks, THREADS_PER_BLOCK>>>(d_in, d_blockMax, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msFirstPass = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msFirstPass, start, stop));

    // Second pass
    // Only one block needed since blockMax size is small
    int secondPassBlocks = 1;
    CHECK_CUDA(cudaEventRecord(start));
    secondPassMax<<<secondPassBlocks, THREADS_PER_BLOCK>>>(d_blockMax, d_finalMax, blocks);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float msSecondPass = 0;
    CHECK_CUDA(cudaEventElapsedTime(&msSecondPass, start, stop));

    int hostTwoPassMax;
    CHECK_CUDA(cudaMemcpy(&hostTwoPassMax, d_finalMax, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    if (hostSingleMax != hostTwoPassMax) {
        fprintf(stderr, "Error: Results differ! Single: %d, Two-pass: %d\n",
                hostSingleMax, hostTwoPassMax);
    } else {
        printf("Maximum value: %d\n", hostSingleMax);
    }

    printf("Single-kernel atomicMax reduction time: %.3f ms\n", msSingle);
    printf("Two-pass reduction times: first pass = %.3f ms, second pass = %.3f ms, total = %.3f ms\n",
           msFirstPass, msSecondPass, msFirstPass + msSecondPass);

    // Cleanup
    free(h_in);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_globalMax));
    CHECK_CUDA(cudaFree(d_blockMax));
    CHECK_CUDA(cudaFree(d_finalMax));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```