```cuda
/*
 * Compare the bandwidth of pageable vs. pinned (non-pageable) host memory.
 * Perform the timing from the previous exercise.
 * Then, repeat it but allocate the host memory using `cudaMallocHost` instead of `malloc`.
 * Print both bandwidths and observe the difference.
 *
 * Thinking:
 * 1. We want to measure data transfer bandwidth between host and device.
 * 2. Use a large buffer, e.g., 100 MB, to reduce noise.
 * 3. Allocate host memory twice: once with malloc (pageable) and once with cudaMallocHost (pinned).
 * 4. Allocate device memory of the same size once (or twice, but one is enough).
 * 5. Initialize the pageable buffer with some data.
 * 6. Measure transfer time using CUDA events (cudaEventRecord, cudaEventSynchronize).
 * 7. Compute bandwidth = bytes / (time_ms / 1000).
 * 8. Print both bandwidths.
 * 9. Clean up: free host memory (free for pageable, cudaFreeHost for pinned) and device memory (cudaFree).
 * 10. Ensure error checking for CUDA calls.
 * 11. The code is self-contained, uses C style, and can be compiled with nvcc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const size_t SIZE = 100 * 1024 * 1024; // 100 MB
    const size_t N = SIZE / sizeof(float); // number of floats

    // Allocate device memory
    float *d_ptr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_ptr, SIZE));

    // -------- Pageable (malloc) ----------
    float *h_pageable = (float *)malloc(SIZE);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        exit(EXIT_FAILURE);
    }
    // Initialize pageable buffer
    for (size_t i = 0; i < N; ++i) {
        h_pageable[i] = (float)i;
    }

    cudaEvent_t start, stop;
    float milliseconds = 0.0f;

    // Create events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Measure host-to-device transfer for pageable memory
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_pageable, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    double bandwidth_pageable = (double)SIZE / (milliseconds / 1000.0) / (1 << 30); // GB/s

    printf("Pageable (malloc) transfer time: %.3f ms\n", milliseconds);
    printf("Pageable (malloc) bandwidth: %.3f GB/s\n\n", bandwidth_pageable);

    // -------- Pinned (cudaMallocHost) ----------
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_pinned, SIZE));

    // Initialize pinned buffer
    for (size_t i = 0; i < N; ++i) {
        h_pinned[i] = (float)i;
    }

    // Measure host-to-device transfer for pinned memory
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_ptr, h_pinned, SIZE, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    double bandwidth_pinned = (double)SIZE / (milliseconds / 1000.0) / (1 << 30); // GB/s

    printf("Pinned (cudaMallocHost) transfer time: %.3f ms\n", milliseconds);
    printf("Pinned (cudaMallocHost) bandwidth: %.3f GB/s\n\n", bandwidth_pinned);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_ptr));

    return 0;
}
```