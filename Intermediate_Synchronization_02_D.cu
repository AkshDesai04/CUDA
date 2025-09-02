```cuda
/*
Aim of the program: Implement a parallel minimum-finding algorithm.

Thinking:
- The task is to find the minimum value in an array of integers using CUDA.
- We'll use the classic reduction pattern: each block loads a chunk of the array into shared memory,
  then iteratively reduces pairs of elements until one value remains per block.
- The per-block minima are written to an intermediate array.
- If the number of blocks is > 1, we launch a second kernel (or recursively) to reduce the block minima.
- For simplicity, we'll perform a two-stage reduction: first kernel reduces to per-block mins,
  second kernel reduces those mins to a single value.
- We'll include basic error checking macros and handle arbitrary array sizes.
- The host code will allocate memory, initialize the array with random integers, copy to device,
  invoke kernels, copy back the result, and print the minimum.
- Shared memory usage is limited to block size; we assume blockDim.x <= 1024 (max for compute capability 3.0+).
- To keep the example self-contained, we use a fixed block size of 256.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>

#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (err_num=%d) at %s:%d\n", \
                    cudaGetErrorString(err_), err_, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel to compute per-block minima
__global__ void minReduceKernel(const int *d_in, int *d_blockMins, int N)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int localMin = INT_MAX;

    // Load two elements per thread (coalesced)
    if (idx < N) localMin = d_in[idx];
    if (idx + blockDim.x < N) {
        int val = d_in[idx + blockDim.x];
        if (val < localMin) localMin = val;
    }

    sdata[tid] = localMin;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }

    // Write block's minimum to global memory
    if (tid == 0) d_blockMins[blockIdx.x] = sdata[0];
}

// Host helper to perform reduction iteratively until single value
int reduceMin(int *d_in, int N)
{
    int *d_intermediate = NULL;
    int currentSize = N;
    int *d_current = d_in;

    while (currentSize > 1) {
        int threadsPerBlock = 256;
        int blocks = (currentSize + threadsPerBlock * 2 - 1) / (threadsPerBlock * 2);

        // Allocate array for block minima
        CUDA_CHECK(cudaMalloc((void**)&d_intermediate, blocks * sizeof(int)));

        size_t sharedMemSize = threadsPerBlock * sizeof(int);
        minReduceKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_current, d_intermediate, currentSize);
        CUDA_CHECK(cudaGetLastError());

        // Prepare for next iteration
        if (d_current != d_in) {
            CUDA_CHECK(cudaFree(d_current));
        }
        d_current = d_intermediate;
        d_intermediate = NULL;

        currentSize = blocks;
    }

    int minVal;
    CUDA_CHECK(cudaMemcpy(&minVal, d_current, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_current));
    return minVal;
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    int *h_data = (int*)malloc(size);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < N; ++i) {
        h_data[i] = rand() % 10000; // values between 0 and 9999
    }

    int *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Compute minimum on GPU
    int gpuMin = reduceMin(d_data, N);
    printf("GPU Minimum: %d\n", gpuMin);

    // Verify with CPU
    int cpuMin = INT_MAX;
    for (int i = 0; i < N; ++i) {
        if (h_data[i] < cpuMin) cpuMin = h_data[i];
    }
    printf("CPU Minimum: %d\n", cpuMin);

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
```