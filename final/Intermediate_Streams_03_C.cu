/*
Try changing the 'work' done by the kernel. If the kernel is very fast, the speedup will be high. If the kernel is very slow (compute-bound), the speedup will be lower because the copies finish long before the kernel does.

Thinking process:
1. The aim is to demonstrate how varying the amount of work performed in a CUDA kernel affects the overall speedup when overlapping memory copies with computation.
2. We need a small CUDA program that:
   - Allocates device and host buffers.
   - Copies data to device asynchronously.
   - Launches a kernel that performs a user‑configurable amount of work per element (e.g., a loop of floating‑point operations).
   - Copies the result back asynchronously.
   - Uses CUDA streams to allow overlap between transfer and kernel execution.
   - Measures the time for:
     a) pure transfer (copy only),
     b) transfer + kernel with minimal work,
     c) transfer + kernel with heavier work.
   - Computes and prints speedups.
3. To keep the code self‑contained, we’ll use `cudaEvent` for timing and simple command‑line arguments to adjust the work factor.
4. The kernel will use a simple arithmetic loop that can be made expensive by increasing the number of iterations (`WORK_FACTOR`).
5. We’ll include error checking, resource cleanup, and print results in a clear format.
6. The program will be compiled as a `.cu` file.

This approach satisfies the requirement to change the "work" done by the kernel and to observe the impact on speedup. It also follows the developer instruction to put the aim and thoughts inside a multiline comment at the top, then provide the full code afterwards.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <chrono>

/* Simple CUDA error checker */
#define CUDA_CHECK(err)                                 \
    do {                                                \
        cudaError_t err__ = (err);                      \
        if (err__ != cudaSuccess) {                    \
            fprintf(stderr, "CUDA error: %s (%d)\n",    \
                    cudaGetErrorString(err__), (int)err__); \
            exit(EXIT_FAILURE);                        \
        }                                               \
    } while (0)

/* Kernel that performs a configurable amount of work per element */
__global__ void compute_kernel(float *d_out, const float *d_in, int N, int work_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float val = d_in[idx];
    /* Perform some dummy work */
    for (int i = 0; i < work_factor; ++i) {
        val = val * 1.0000001f + 0.0000001f;
    }
    d_out[idx] = val;
}

/* Helper to measure elapsed time in milliseconds using chrono */
double get_elapsed_ms(const std::chrono::high_resolution_clock::time_point &start,
                      const std::chrono::high_resolution_clock::time_point &end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char *argv[]) {
    /* Default parameters */
    size_t N = 1 << 24;              // Number of elements (~16M)
    int work_factor_fast = 1;        // Minimal work
    int work_factor_slow = 100000;   // Heavy work

    if (argc > 1) N = static_cast<size_t>(atoll(argv[1]));
    if (argc > 2) work_factor_fast = atoi(argv[2]);
    if (argc > 3) work_factor_slow = atoi(argv[3]);

    size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    for (size_t i = 0; i < N; ++i) h_in[i] = 1.0f;  // Dummy data

    /* Allocate device memory */
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    /* Create CUDA stream for async operations */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* Events for timing */
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    /* 1. Pure transfer timing (async copy both ways) */
    auto t1 = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t2 = std::chrono::high_resolution_clock::now();
    double transfer_time_ms = get_elapsed_ms(t1, t2);
    printf("Pure transfer time (ms): %f\n", transfer_time_ms);

    /* 2. Transfer + kernel with fast work */
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    compute_kernel<<<(N + 255)/256, 256, 0, stream>>>(d_out, d_in, N, work_factor_fast);
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float fast_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&fast_total_ms, startEvent, stopEvent));
    double speedup_fast = transfer_time_ms / fast_total_ms;
    printf("Transfer + kernel (fast work) total time (ms): %f, speedup: %f\n", fast_total_ms, speedup_fast);

    /* 3. Transfer + kernel with slow work */
    CUDA_CHECK(cudaEventRecord(startEvent, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    compute_kernel<<<(N + 255)/256, 256, 0, stream>>>(d_out, d_in, N, work_factor_slow);
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(stopEvent, stream));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float slow_total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&slow_total_ms, startEvent, stopEvent));
    double speedup_slow = transfer_time_ms / slow_total_ms;
    printf("Transfer + kernel (slow work) total time (ms): %f, speedup: %f\n", slow_total_ms, speedup_slow);

    /* Cleanup */
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
