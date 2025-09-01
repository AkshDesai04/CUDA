```cuda
/*
Compare the bandwidth of pageable vs. pinned (non-pageable) host memory. Perform the timing from the previous exercise. Then, repeat it but allocate the host memory using cudaMallocHost instead of malloc. Print both bandwidths and observe the difference.

Thinking:
1. The goal is to measure the difference in memory transfer performance between pageable host memory (obtained via standard malloc) and pinned host memory (obtained via cudaMallocHost).
2. For a fair comparison we allocate two host buffers of identical size, one pageable and one pinned. We also allocate a device buffer large enough to hold the data.
3. We will perform two copy operations for each host buffer: host-to-device (H2D) and device-to-host (D2H). For each operation we time the transfer using CUDA events.
4. Bandwidth is computed as (bytes transferred) / (time in seconds). Since cudaEventElapsedTime returns milliseconds, we convert to seconds.
5. We choose a buffer size that is large enough to amortize overhead but small enough to fit in typical GPU memory. 64 MiB of floats (16M floats) is a safe choice; each transfer is 64 MiB, so the GPU memory usage is about 64 MiB for the device buffer.
6. The program prints the bandwidth for pageable H2D, pageable D2H, pinned H2D, and pinned D2H. We expect pinned transfers to be significantly faster.
7. Error checking is performed via a macro to keep the code clean. All allocated memory is freed before program exit.
8. The program uses only standard CUDA API calls and should compile with nvcc.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Number of floats: 16M floats -> 64 MiB
#define NUM_FLOATS (16 * 1024 * 1024)
#define NUM_BYTES (NUM_FLOATS * sizeof(float))

int main(void) {
    // Allocate pageable host memory
    float *h_pageable = (float *)malloc(NUM_BYTES);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate pinned host memory
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_pinned, NUM_BYTES));

    // Allocate device memory
    float *d_buf = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_buf, NUM_BYTES));

    // Initialize host buffers
    for (size_t i = 0; i < NUM_FLOATS; ++i) {
        h_pageable[i] = (float)i;
        h_pinned[i]   = (float)i;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float h2d_time_ms = 0.0f;
    float d2h_time_ms = 0.0f;

    // ---- Pageable host memory ----
    // Host-to-Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_buf, h_pageable, NUM_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, start, stop));

    // Device-to-Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pageable, d_buf, NUM_BYTES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time_ms, start, stop));

    double pageable_h2d_bw = (NUM_BYTES / (h2d_time_ms * 1e-3)) / 1e9; // GB/s
    double pageable_d2h_bw = (NUM_BYTES / (d2h_time_ms * 1e-3)) / 1e9; // GB/s

    // ---- Pinned host memory ----
    // Host-to-Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_buf, h_pinned, NUM_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, start, stop));

    // Device-to-Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pinned, d_buf, NUM_BYTES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time_ms, start, stop));

    double pinned_h2d_bw = (NUM_BYTES / (h2d_time_ms * 1e-3)) / 1e9; // GB/s
    double pinned_d2h_bw = (NUM_BYTES / (d2h_time_ms * 1e-3)) / 1e9; // GB/s

    // Output results
    printf("Pageable Host Memory:\n");
    printf("  Host-to-Device Bandwidth: %.2f GB/s\n", pageable_h2d_bw);
    printf("  Device-to-Host Bandwidth: %.2f GB/s\n", pageable_d2h_bw);

    printf("\nPinned Host Memory (cudaMallocHost):\n");
    printf("  Host-to-Device Bandwidth: %.2f GB/s\n", pinned_h2d_bw);
    printf("  Device-to-Host Bandwidth: %.2f GB/s\n", pinned_d2h_bw);

    // Clean up
    free(h_pageable);
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```