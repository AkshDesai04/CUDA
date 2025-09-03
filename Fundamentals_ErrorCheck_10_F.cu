/*
Create two timers to measure two different parts of your program sequentially.
I wanted to demonstrate how to time different sections of a CUDA program using
both GPU events and CPU timers.  The idea is to launch two simple kernels
(one that does a dummy vector addition, the other that does a slightly
different operation) and measure the elapsed time of each kernel separately.
For the GPU timing I use cudaEvent_t, recording before and after each
kernel and then synchronizing and computing the elapsed time.  For the CPU
side I also show how to use std::chrono to measure the wall‑clock time
between two points in the host code, which can be useful if you want to
time host work that is not on the GPU.  The program allocates a pair of
device buffers, runs each kernel, measures the time, copies results back
to the host, and prints the timings.  All CUDA API calls are wrapped in
a simple error checking macro so that any failures will print a message
and abort.  The code is self‑contained, can be compiled with nvcc, and
illustrates two distinct timers measuring two different parts of a
sequential program flow.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Simple vector addition kernel (part 1)
__global__ void kernel_add(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

// Simple vector subtraction kernel (part 2)
__global__ void kernel_sub(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] - b[idx];
}

int main() {
    const int N = 1 << 20;            // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start1, stop1, start2, stop2;
    CUDA_CHECK(cudaEventCreate(&start1));
    CUDA_CHECK(cudaEventCreate(&stop1));
    CUDA_CHECK(cudaEventCreate(&start2));
    CUDA_CHECK(cudaEventCreate(&stop2));

    // ---------- First part: kernel_add ----------
    CUDA_CHECK(cudaEventRecord(start1, 0));
    kernel_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop1, 0));
    CUDA_CHECK(cudaEventSynchronize(stop1));

    float ms_part1 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_part1, start1, stop1));

    // Copy result back to host for sanity check
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Optional: verify result of addition
    bool correct_add = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
            correct_add = false;
            break;
        }
    }

    // ---------- Second part: kernel_sub ----------
    CUDA_CHECK(cudaEventRecord(start2, 0));
    kernel_sub<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaEventRecord(stop2, 0));
    CUDA_CHECK(cudaEventSynchronize(stop2));

    float ms_part2 = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_part2, start2, stop2));

    // Copy result back to host for sanity check
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // Optional: verify result of subtraction
    bool correct_sub = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_c[i] - (h_a[i] - h_b[i])) > 1e-5f) {
            correct_sub = false;
            break;
        }
    }

    // ---------- Host-side timing example ----------
    auto host_start = std::chrono::high_resolution_clock::now();

    // Simulate some host work: simple loop
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += h_a[i];

    auto host_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> host_ms = host_stop - host_start;

    // ---------- Output ----------
    printf("GPU timing:\n");
    printf("  kernel_add elapsed time: %f ms (%s)\n", ms_part1, correct_add ? "PASS" : "FAIL");
    printf("  kernel_sub elapsed time: %f ms (%s)\n", ms_part2, correct_sub ? "PASS" : "FAIL");
    printf("Host timing:\n");
    printf("  Simulated host work elapsed time: %f ms\n", host_ms.count());

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start1));
    CUDA_CHECK(cudaEventDestroy(stop1));
    CUDA_CHECK(cudaEventDestroy(start2));
    CUDA_CHECK(cudaEventDestroy(stop2));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
