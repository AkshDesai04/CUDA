/*
Compare the bandwidth of pageable vs. pinned (non-pageable) host memory. Perform the timing from the previous exercise. Then, repeat it but allocate the host memory using `cudaMallocHost` instead of `malloc`. Print both bandwidths and observe the difference.

Thinking process:
1. The goal is to measure transfer speeds between host and device for two types of host memory: normal pageable memory (allocated with `malloc`) and pinned memory (allocated with `cudaMallocHost`). 
2. Use `cudaEvent_t` for accurate timing of memory copies because it measures GPU activity directly.
3. For each memory type:
   - Allocate host buffer (pageable or pinned).
   - Allocate device buffer with `cudaMalloc`.
   - Perform a host-to-device copy and a device-to-host copy, timing each operation.
   - Compute bandwidth as (bytes transferred / time in seconds). Typically bandwidth is reported in GB/s. We'll calculate two bandwidths: one for H2D and one for D2H, and then take the average to get a representative number.
4. After measuring both memory types, print the results clearly so that differences can be observed.
5. Add basic error checking for CUDA API calls via a helper macro.
6. Use a moderate buffer size (e.g., 100 MB) to avoid exhausting system memory while still giving a noticeable difference.
7. Ensure all resources are freed (`cudaFree`, `free`, `cudaFreeHost`) before exiting.

Potential pitfalls:
- Forgetting to synchronize before reading the elapsed time could give inaccurate results.
- Using `cudaEventElapsedTime` returns time in milliseconds, so conversion to seconds is required.
- `cudaMallocHost` must be paired with `cudaFreeHost`.
- The host memory must be large enough to fill the transfer; otherwise, the GPU might be idle.
- Printing should convert bytes to gigabytes for readability.

With these considerations, the following CUDA C code implements the measurement and prints the bandwidths for both pageable and pinned host memory.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),        \
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const size_t N_BYTES = 100 * 1024 * 1024; // 100 MB
    const size_t N_ELEMS = N_BYTES / sizeof(float);

    // ====================== PAGEABLE MEMORY ======================
    float *h_pageable = (float *)malloc(N_BYTES);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host memory to avoid lazy allocation issues
    for (size_t i = 0; i < N_ELEMS; ++i) {
        h_pageable[i] = (float)i;
    }

    float *d_buf = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_buf, N_BYTES));

    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Host to Device transfer timing
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_buf, h_pageable, N_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float h2d_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, start, stop));

    // Device to Host transfer timing
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pageable, d_buf, N_BYTES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float d2h_time_ms;
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time_ms, start, stop));

    // Calculate bandwidths (GB/s)
    double h2d_bandwidth = (N_BYTES / (1024.0 * 1024.0 * 1024.0)) / (h2d_time_ms / 1000.0);
    double d2h_bandwidth = (N_BYTES / (1024.0 * 1024.0 * 1024.0)) / (d2h_time_ms / 1000.0);
    double pageable_avg = (h2d_bandwidth + d2h_bandwidth) / 2.0;

    printf("Pageable memory:\n");
    printf("  H2D: %.2f GB/s\n", h2d_bandwidth);
    printf("  D2H: %.2f GB/s\n", d2h_bandwidth);
    printf("  Avg : %.2f GB/s\n", pageable_avg);

    // Clean up pageable resources
    CHECK_CUDA(cudaFree(d_buf));
    free(h_pageable);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // ====================== PINNED MEMORY ======================
    float *h_pinned = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_pinned, N_BYTES)); // pinned memory

    // Initialize pinned memory
    for (size_t i = 0; i < N_ELEMS; ++i) {
        h_pinned[i] = (float)i;
    }

    // Allocate device buffer again
    CHECK_CUDA(cudaMalloc((void **)&d_buf, N_BYTES));

    // Create new events
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // H2D timing
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_buf, h_pinned, N_BYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&h2d_time_ms, start, stop));

    // D2H timing
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pinned, d_buf, N_BYTES, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&d2h_time_ms, start, stop));

    // Bandwidth calculations
    h2d_bandwidth = (N_BYTES / (1024.0 * 1024.0 * 1024.0)) / (h2d_time_ms / 1000.0);
    d2h_bandwidth = (N_BYTES / (1024.0 * 1024.0 * 1024.0)) / (d2h_time_ms / 1000.0);
    double pinned_avg = (h2d_bandwidth + d2h_bandwidth) / 2.0;

    printf("\nPinned memory:\n");
    printf("  H2D: %.2f GB/s\n", h2d_bandwidth);
    printf("  D2H: %.2f GB/s\n", d2h_bandwidth);
    printf("  Avg : %.2f GB/s\n", pinned_avg);

    // Clean up pinned resources
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
