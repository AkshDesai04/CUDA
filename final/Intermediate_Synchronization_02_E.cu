```cuda
/*
Aim: Compare the performance of your two-pass reduction with Thrust's thrust::reduce or thrust::max_element.

Thinking process:
1. Define the problem: we need a CUDA program that performs a reduction (sum and max) in two passes
   – first a kernel that reduces blocks of the input array into partial results, then a second step
   – that aggregates those partial results on the host (for simplicity). We will measure the GPU
     time of the two-pass approach.
2. We also need to perform the same reduction using Thrust's high-level functions:
   - thrust::reduce for summation
   - thrust::max_element for maximum.
   Timing must be comparable, so we wrap the Thrust calls between cudaEvent markers and call
   cudaDeviceSynchronize() before reading the elapsed time.
3. For random data generation, use std::mt19937 on the host, fill an array of size N (e.g. 2^24)
   with int values.
4. Allocate device memory for the input array and a partial results array. Copy the host data to the
   device once.
5. Implement two kernels:
   - sumReduceKernel: uses shared memory, each thread loads up to two elements, reduces within the
     block, writes block result to partial array.
   - maxReduceKernel: similar but uses the max operator, initialized with INT_MIN.
6. The number of blocks is computed based on the fact that each block processes 2*BLOCK_SIZE elements.
7. After launching the first kernel, we copy the partial results back to the host and perform the
   final reduction there. This keeps the code simple and avoids a second kernel.
8. For Thrust, wrap the device pointer in thrust::device_ptr and call the reduce and max_element
   functions directly. We time these calls with cudaEvent.
9. Finally, print out the results and the elapsed times for each approach so the user can compare.
10. Add error checking macros for CUDA calls.

Edge cases considered:
- When the input size isn't an exact multiple of block size, we clamp indices.
- We use int32 for simplicity; for larger data types, change accordingly.
- We use shared memory of size BLOCK_SIZE for both kernels.

Compile with:
   nvcc -O2 -o reduction_comparison reduction_comparison.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <random>
#include <chrono>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

#define BLOCK_SIZE 256

// Two-pass sum reduction kernel
__global__ void sumReduceKernel(const int *input, int *partial, int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;
    if (idx < n) sum += input[idx];
    if (idx + blockDim.x < n) sum += input[idx + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // In-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// Two-pass max reduction kernel
__global__ void maxReduceKernel(const int *input, int *partial, int n)
{
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int val = INT_MIN;
    if (idx < n) val = input[idx];
    if (idx + blockDim.x < n) val = max(val, input[idx + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    // In-block reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = max(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

int main()
{
    const size_t N = 1 << 24;          // 16M elements
    const size_t partialCount = (N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2);

    // Host data
    int *h_data = (int*)malloc(N * sizeof(int));
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 1000);
    for (size_t i = 0; i < N; ++i) h_data[i] = dist(rng);

    // Device data
    int *d_data, *d_partial;
    CHECK_CUDA(cudaMalloc((void**)&d_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_partial, partialCount * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice));

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Two-pass sum reduction
    float msTwoPassSum = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    sumReduceKernel<<<partialCount, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, d_partial, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msTwoPassSum, start, stop));

    // Copy partials to host and finalize sum
    int *h_partial = (int*)malloc(partialCount * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_partial, d_partial, partialCount * sizeof(int), cudaMemcpyDeviceToHost));
    int sumTwoPass = 0;
    for (size_t i = 0; i < partialCount; ++i) sumTwoPass += h_partial[i];

    // Two-pass max reduction
    float msTwoPassMax = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    maxReduceKernel<<<partialCount, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, d_partial, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msTwoPassMax, start, stop));

    int maxTwoPass = INT_MIN;
    for (size_t i = 0; i < partialCount; ++i) maxTwoPass = max(maxTwoPass, h_partial[i]);

    // Thrust reduce (sum)
    float msThrustReduce = 0.0f;
    thrust::device_ptr<int> dev_ptr(d_data);
    CHECK_CUDA(cudaEventRecord(start));
    int sumThrust = thrust::reduce(dev_ptr, dev_ptr + N, 0, thrust::plus<int>());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msThrustReduce, start, stop));

    // Thrust max_element
    float msThrustMax = 0.0f;
    CHECK_CUDA(cudaEventRecord(start));
    int maxThrust = *thrust::max_element(dev_ptr, dev_ptr + N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&msThrustMax, start, stop));

    // Print results
    printf("=== Two-pass Reduction ===\n");
    printf("Sum: %d, Time: %.3f ms\n", sumTwoPass, msTwoPassSum);
    printf("Max: %d, Time: %.3f ms\n", maxTwoPass, msTwoPassMax);

    printf("\n=== Thrust Reduction ===\n");
    printf("Sum: %d, Time: %.3f ms\n", sumThrust, msThrustReduce);
    printf("Max: %d, Time: %.3f ms\n", maxThrust, msThrustMax);

    // Clean up
    free(h_data);
    free(h_partial);
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_partial));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```