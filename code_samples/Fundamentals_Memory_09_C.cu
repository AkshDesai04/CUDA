```cuda
/*
Time both HtoD and DtoH transfers with pinned memory and compare to the pageable case.

Thought process:
1. The goal is to measure and compare the performance of memory transfers
   between host and device for two types of host memory:
   - Pageable memory (allocated with malloc)
   - Pinned (page-locked) memory (allocated with cudaMallocHost)

2. For each case, we will:
   a. Allocate host memory.
   b. Allocate device memory.
   c. Time Host-to-Device (HtoD) transfer.
   d. Time Device-to-Host (DtoH) transfer.
   e. Clean up.

3. Timing will be performed using CUDA events (cudaEventRecord / cudaEventElapsedTime),
   which give time in milliseconds between two events on the GPU.

4. We'll use a reasonably large buffer (e.g., 100 MB) to make transfer times noticeable.
   100 MB = 100 * 1024 * 1024 bytes.

5. We will implement a simple error-checking macro `CUDA_CHECK` to catch any CUDA API errors.

6. The program will print the elapsed time for each transfer and each memory type,
   along with the transfer bandwidth computed as data size / time.

7. No external libraries beyond CUDA Runtime are needed. The code is self-contained
   and can be compiled with `nvcc` into an executable.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                        \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void) {
    const size_t num_bytes = 100 * 1024 * 1024; // 100 MB
    const size_t num_elems = num_bytes / sizeof(float);

    // Pointers for pageable host memory
    float *h_pageable = NULL;
    // Pointers for pinned host memory
    float *h_pinned = NULL;
    // Device memory
    float *d_mem = NULL;

    // Allocate pageable host memory
    h_pageable = (float*)malloc(num_bytes);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate pinned host memory
    CHECK_CUDA(cudaMallocHost((void**)&h_pinned, num_bytes));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_mem, num_bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Variables to hold elapsed times
    float htoD_pageable_ms = 0.0f;
    float dtoH_pageable_ms = 0.0f;
    float htoD_pinned_ms  = 0.0f;
    float dtoH_pinned_ms  = 0.0f;

    // ----------------------------
    // Pageable memory transfers
    // ----------------------------
    // Host-to-Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_mem, h_pageable, num_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&htoD_pageable_ms, start, stop));

    // Device-to-Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pageable, d_mem, num_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&dtoH_pageable_ms, start, stop));

    // ----------------------------
    // Pinned memory transfers
    // ----------------------------
    // Host-to-Device
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(d_mem, h_pinned, num_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&htoD_pinned_ms, start, stop));

    // Device-to-Host
    CHECK_CUDA(cudaEventRecord(start, 0));
    CHECK_CUDA(cudaMemcpy(h_pinned, d_mem, num_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&dtoH_pinned_ms, start, stop));

    // Compute bandwidths (GB/s)
    double bandwidth_pageable_htod = (double)num_bytes / (htoD_pageable_ms / 1000.0) / (1 << 30);
    double bandwidth_pageable_dtoh = (double)num_bytes / (dtoH_pageable_ms / 1000.0) / (1 << 30);
    double bandwidth_pinned_htod  = (double)num_bytes / (htoD_pinned_ms  / 1000.0) / (1 << 30);
    double bandwidth_pinned_dtoh  = (double)num_bytes / (dtoH_pinned_ms  / 1000.0) / (1 << 30);

    // Print results
    printf("Transfer timings for %zu MB (%.2f GB) buffer:\n", num_bytes / (1024 * 1024), (double)num_bytes / (1ULL << 30));
    printf("\nPageable memory:\n");
    printf("  Host -> Device: %.3f ms  (%.2f GB/s)\n", htoD_pageable_ms, bandwidth_pageable_htod);
    printf("  Device -> Host: %.3f ms  (%.2f GB/s)\n", dtoH_pageable_ms, bandwidth_pageable_dtoh);

    printf("\nPinned memory:\n");
    printf("  Host -> Device: %.3f ms  (%.2f GB/s)\n", htoD_pinned_ms, bandwidth_pinned_htod);
    printf("  Device -> Host: %.3f ms  (%.2f GB/s)\n", dtoH_pinned_ms, bandwidth_pinned_dtoh);

    // Clean up
    CHECK_CUDA(cudaFree(d_mem));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    free(h_pageable);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```