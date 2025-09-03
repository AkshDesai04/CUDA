/*
 * Aim: Time the naive (racy) kernel vs. the atomicAdd kernel. Atomic operations have a performance cost. Is it significant here?
 *
 * Thinking:
 * 1. Create two kernels: naive_kernel performs `*counter += 1` inside a loop; atomic_kernel performs `atomicAdd(counter, 1ULL)`.
 * 2. Use a single-element counter array on device. Each thread loops ITER times, incrementing the counter.
 * 3. Launch a large number of threads (N) to stress the counter update. Use N=1<<20 (1M) and ITER=1000, giving 1e9 increments.
 * 4. Measure execution time with cudaEvent_t for each kernel.
 * 5. Copy back the result to host and print. Naive kernel result will be lower due to lost updates. Atomic kernel should match N*ITER.
 * 6. Compare the timings to see overhead of atomicAdd. Also print ratio.
 * 7. Include basic error checking macro.
 * 8. Use unsigned long long for counter to avoid overflow.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

__global__ void naive_kernel(unsigned long long *counter, int N, int ITER)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        for (int i = 0; i < ITER; ++i)
        {
            // Racy increment: read-modify-write without synchronization
            unsigned long long temp = *counter;
            temp += 1ULL;
            *counter = temp;
        }
    }
}

__global__ void atomic_kernel(unsigned long long *counter, int N, int ITER)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        for (int i = 0; i < ITER; ++i)
        {
            // Atomic increment to avoid race conditions
            atomicAdd(counter, 1ULL);
        }
    }
}

int main(void)
{
    const int N = 1 << 20;      // Number of threads (1M)
    const int ITER = 1000;      // Number of increments per thread
    const int THREADS_PER_BLOCK = 256;
    const int NUM_BLOCKS = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    unsigned long long *d_counter = NULL;
    unsigned long long h_counter = 0ULL;
    size_t counter_bytes = sizeof(unsigned long long);

    // Allocate device counter and initialize to zero
    CUDA_CHECK(cudaMalloc((void**)&d_counter, counter_bytes));
    CUDA_CHECK(cudaMemset(d_counter, 0, counter_bytes));

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---- Naive kernel ----
    CUDA_CHECK(cudaEventRecord(start, 0));
    naive_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_counter, N, ITER);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, counter_bytes, cudaMemcpyDeviceToHost));
    printf("Naive kernel result: %llu (expected %llu)\n", h_counter, (unsigned long long)N * ITER);
    printf("Naive kernel time: %f ms\n", ms_naive);

    // Reset counter
    CUDA_CHECK(cudaMemset(d_counter, 0, counter_bytes));

    // ---- Atomic kernel ----
    CUDA_CHECK(cudaEventRecord(start, 0));
    atomic_kernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_counter, N, ITER);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_atomic = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_atomic, start, stop));

    CUDA_CHECK(cudaMemcpy(&h_counter, d_counter, counter_bytes, cudaMemcpyDeviceToHost));
    printf("Atomic kernel result: %llu (expected %llu)\n", h_counter, (unsigned long long)N * ITER);
    printf("Atomic kernel time: %f ms\n", ms_atomic);

    // Ratio
    printf("Atomic time / Naive time: %f\n", ms_atomic / ms_naive);

    // Cleanup
    CUDA_CHECK(cudaFree(d_counter));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
