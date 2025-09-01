```cuda
/*
An alternative to loading 0 is to make the computation loop conditional. Is this more or less efficient?

My thinking:
  The question asks whether using a conditional to skip a loop (when the loop body is not needed)
  is more or less efficient than the traditional approach of loading an initial zero value
  and always executing the loop body.  The trade‑off depends on:
    • The cost of the conditional branch itself (branch misprediction, divergence).
    • The cost of the loop iterations (memory loads, arithmetic).
    • The fraction of threads for which the loop is skipped.

  In CUDA, divergent branches among threads in a warp can serialize execution, so a conditional
  that many threads evaluate to false can lead to poor utilization.  On the other hand, if the
  loop is rarely executed, the overhead of looping (condition checks, memory traffic) dominates,
  so skipping the loop entirely may be cheaper.

  To illustrate this, the following CUDA program benchmarks two kernels:
    1. `compute_with_load0`: loads a zero accumulator and always executes a loop over a
       per‑thread array of length `len`.  The loop runs zero times if `len==0`, but the loop
       construct is still present.
    2. `compute_conditional_loop`: uses an `if (len > 0)` guard to skip the loop entirely
       when `len==0`.

  The program runs both kernels under two scenarios:
    a) All threads have `len==0` (loop never executes).
    b) All threads have a non‑zero `len` (loop executes fully).

  By timing both kernels in these scenarios we can observe when the conditional approach
  is faster or slower.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that always performs the loop, even when len == 0
__global__ void compute_with_load0(const float* data, int* out, const int* lens, int maxlen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int len = lens[tid];
    float sum = 0.0f;                     // load zero into accumulator

    // The loop exists regardless of len; if len == 0, the body never executes
    for (int i = 0; i < len; ++i)
    {
        sum += data[tid * maxlen + i];
    }
    out[tid] = (int)sum; // just store some result
}

// Kernel that conditionally executes the loop only when len > 0
__global__ void compute_conditional_loop(const float* data, int* out, const int* lens, int maxlen)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int len = lens[tid];
    float sum = 0.0f;

    if (len > 0)
    {
        for (int i = 0; i < len; ++i)
        {
            sum += data[tid * maxlen + i];
        }
    }
    out[tid] = (int)sum;
}

void check_cuda_error(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error after %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    const int threadsPerBlock = 256;
    const int numBlocks = 64;
    const int N = threadsPerBlock * numBlocks;     // total threads
    const int maxlen = 1024;                       // max elements per thread

    size_t totalElements = (size_t)N * maxlen;

    // Allocate host memory
    float* h_data = (float*)malloc(totalElements * sizeof(float));
    int*   h_lens = (int*)malloc(N * sizeof(int));
    int*   h_out  = (int*)malloc(N * sizeof(int));

    // Initialize data
    for (size_t i = 0; i < totalElements; ++i)
        h_data[i] = 1.0f;          // simple value

    // Host device buffers
    float* d_data = nullptr;
    int*   d_lens = nullptr;
    int*   d_out  = nullptr;

    cudaMalloc((void**)&d_data, totalElements * sizeof(float));
    cudaMalloc((void**)&d_lens, N * sizeof(int));
    cudaMalloc((void**)&d_out,  N * sizeof(int));

    cudaMemcpy(d_data, h_data, totalElements * sizeof(float), cudaMemcpyHostToDevice);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Scenario A: all lens == 0 (loop never executes)
    for (int i = 0; i < N; ++i) h_lens[i] = 0;
    cudaMemcpy(d_lens, h_lens, N * sizeof(int), cudaMemcpyHostToDevice);

    // Measure compute_with_load0
    cudaEventRecord(start);
    compute_with_load0<<<numBlocks, threadsPerBlock>>>(d_data, d_out, d_lens, maxlen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_load0_a = 0;
    cudaEventElapsedTime(&time_load0_a, start, stop);

    // Measure compute_conditional_loop
    cudaEventRecord(start);
    compute_conditional_loop<<<numBlocks, threadsPerBlock>>>(d_data, d_out, d_lens, maxlen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_cond_a = 0;
    cudaEventElapsedTime(&time_cond_a, start, stop);

    printf("Scenario A (len == 0 for all threads):\n");
    printf("  compute_with_load0 : %f ms\n", time_load0_a);
    printf("  compute_conditional_loop : %f ms\n", time_cond_a);

    // Scenario B: all lens == maxlen (loop executes fully)
    for (int i = 0; i < N; ++i) h_lens[i] = maxlen;
    cudaMemcpy(d_lens, h_lens, N * sizeof(int), cudaMemcpyHostToDevice);

    // Measure compute_with_load0
    cudaEventRecord(start);
    compute_with_load0<<<numBlocks, threadsPerBlock>>>(d_data, d_out, d_lens, maxlen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_load0_b = 0;
    cudaEventElapsedTime(&time_load0_b, start, stop);

    // Measure compute_conditional_loop
    cudaEventRecord(start);
    compute_conditional_loop<<<numBlocks, threadsPerBlock>>>(d_data, d_out, d_lens, maxlen);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_cond_b = 0;
    cudaEventElapsedTime(&time_cond_b, start, stop);

    printf("\nScenario B (len == maxlen for all threads):\n");
    printf("  compute_with_load0 : %f ms\n", time_load0_b);
    printf("  compute_conditional_loop : %f ms\n", time_cond_b);

    // Clean up
    cudaFree(d_data);
    cudaFree(d_lens);
    cudaFree(d_out);
    free(h_data);
    free(h_lens);
    free(h_out);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```