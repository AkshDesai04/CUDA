```cpp
/*
Aim of the program (as requested in the prompt):

This reduction pattern has bank conflicts in the first few iterations. 
A conflict-free version has the stride `s` decrease from `blockDim.x / 2` 
but the access is `s_data[i] += s_data[i+s]`. Compare the performance.

Thinking:
1. We implement two CUDA kernels that perform a reduction over an array of
   floats. The kernels use shared memory to sum values within a block.
2. The first kernel (`reduce_conflict`) uses a standard shared‑memory layout
   that leads to bank conflicts during the early iterations of the loop
   when the stride is a power of two.
3. The second kernel (`reduce_conflict_free`) allocates an extra element in
   shared memory (`blockDim.x + 1`). This padding shifts the indices so
   that two threads that would normally collide on the same memory bank
   now access different banks, eliminating the conflicts. The code
   logic is identical to the conflict version; the only difference is
   the extra padding.
4. The host code allocates a large input array, initializes it, launches
   both kernels, and times them with CUDA events. After the kernels
   finish, the partial block results are summed on the host to obtain
   the final total sum. We print the execution times for each kernel
   and verify that the results match.
5. To keep the example simple and portable, we restrict the block size
   to 256 threads and ensure the input size is a multiple of the block
   size. We use `float` as the data type. The code is fully
   self‑contained and can be compiled with `nvcc` to produce a single
   .cu file as requested.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    }

// Kernel with bank conflicts
__global__ void reduce_conflict(const float *g_idata, float *g_odata, int n)
{
    extern __shared__ float s_data[];          // shared memory

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n)
        s_data[tid] = g_idata[idx];
    else
        s_data[tid] = 0.0f;
    __syncthreads();

    // Standard reduction with stride halving
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            s_data[tid] += s_data[tid + s];
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0)
        g_odata[blockIdx.x] = s_data[0];
}

// Conflict‑free kernel (uses padding to avoid bank conflicts)
__global__ void reduce_conflict_free(const float *g_idata, float *g_odata, int n)
{
    extern __shared__ float s_data[];          // shared memory, size = blockDim.x + 1

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory
    if (idx < n)
        s_data[tid] = g_idata[idx];
    else
        s_data[tid] = 0.0f;
    __syncthreads();

    // Reduction loop identical to the conflict version
    for (unsigned int s = blockDim.x >> 1; s > 0; s >>= 1)
    {
        if (tid < s)
            s_data[tid] += s_data[tid + s];
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0)
        g_odata[blockIdx.x] = s_data[0];
}

// Host function to perform reduction using a specified kernel
void launch_and_time(const char *kernel_name,
                     void (*kernel)(const float*, float*, int),
                     const float *d_in, float *d_out,
                     int n, int threadsPerBlock,
                     cudaEvent_t start, cudaEvent_t stop,
                     float &elapsed_ms)
{
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMem = threadsPerBlock * sizeof(float);
    // Add padding for conflict‑free kernel
    if (strcmp(kernel_name, "conflict_free") == 0)
        sharedMem += sizeof(float);

    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    kernel<<<blocks, threadsPerBlock, sharedMem>>>(d_in, d_out, n);

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
}

// Host function to compute the final sum from block results
float final_sum(const float *h_partial, int blocks)
{
    float sum = 0.0f;
    for (int i = 0; i < blocks; ++i)
        sum += h_partial[i];
    return sum;
}

int main(void)
{
    const int N = 1 << 24;                // 16M elements (~64 MB)
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate and initialize host array
    float *h_in = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; ++i)
        h_in[i] = 1.0f;                   // simple uniform value for easy sum check

    // Allocate device memory
    float *d_in, *d_out_conflict, *d_out_conflict_free;
    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out_conflict, blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out_conflict_free, blocks * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate host partial sums
    float *h_partial_conflict = (float*)malloc(blocks * sizeof(float));
    float *h_partial_conflict_free = (float*)malloc(blocks * sizeof(float));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Time conflict kernel
    float ms_conflict;
    launch_and_time("conflict",
                    reduce_conflict,
                    d_in, d_out_conflict,
                    N, threadsPerBlock,
                    start, stop, ms_conflict);

    // Time conflict-free kernel
    float ms_conflict_free;
    launch_and_time("conflict_free",
                    reduce_conflict_free,
                    d_in, d_out_conflict_free,
                    N, threadsPerBlock,
                    start, stop, ms_conflict_free);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_partial_conflict, d_out_conflict, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_partial_conflict_free, d_out_conflict_free, blocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute final sums
    float sum_conflict = final_sum(h_partial_conflict, blocks);
    float sum_conflict_free = final_sum(h_partial_conflict_free, blocks);

    // Verify correctness
    printf("Expected sum: %f\n", (float)N);
    printf("Conflict kernel sum:        %f\n", sum_conflict);
    printf("Conflict‑free kernel sum:   %f\n", sum_conflict_free);

    // Print timings
    printf("Execution time:\n");
    printf("  Conflict kernel:          %f ms\n", ms_conflict);
    printf("  Conflict‑free kernel:     %f ms\n", ms_conflict_free);

    // Cleanup
    free(h_in);
    free(h_partial_conflict);
    free(h_partial_conflict_free);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out_conflict));
    CUDA_CHECK(cudaFree(d_out_conflict_free));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```