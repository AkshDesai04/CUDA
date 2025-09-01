```cpp
/* 
Aim of the program (word to word from the prompt):
Create a table in your output showing Pageable HtoD, Pinned HtoD, Pageable DtoH, and Pinned DtoH bandwidths.

Thinking:
The task is to write a CUDA C program that measures the bandwidth of four types of memory copies:
1. Pageable Host to Device (HtoD)
2. Pinned Host to Device (HtoD)
3. Pageable Device to Host (DtoH)
4. Pinned Device to Host (DtoH)

To accomplish this, the program will:
- Allocate a buffer of a fixed size (e.g., 64 MB) for the host. It will use both pageable (malloc) and pinned (cudaMallocHost) memory.
- Allocate a buffer of the same size on the device (cudaMalloc).
- Perform a number of copy iterations (e.g., 100) for each copy type, timing each copy with CUDA events to achieve accurate GPU timing.
- Compute the average time per copy and then calculate the bandwidth in GB/s using the formula:
      bandwidth = (size * numIterations) / (elapsedTime / 1e9)
  where elapsedTime is in milliseconds.
- Print a nicely formatted table showing the bandwidths for each copy type.
- Include error checking for all CUDA API calls to ensure robust execution.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main() {
    const size_t KB = 1024;
    const size_t MB = 1024 * KB;
    const size_t GB = 1024 * MB;
    const size_t BUF_SIZE = 64 * MB;  // 64 MB buffer
    const int NUM_ITER = 100;         // Number of iterations for averaging

    // Allocate pageable host memory
    void* h_pageable = malloc(BUF_SIZE);
    if (!h_pageable) {
        fprintf(stderr, "Failed to allocate pageable host memory.\n");
        return EXIT_FAILURE;
    }

    // Allocate pinned host memory
    void* h_pinned = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_pinned, BUF_SIZE));

    // Allocate device memory
    void* d_mem = nullptr;
    CHECK_CUDA(cudaMalloc(&d_mem, BUF_SIZE));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float elapsedTime = 0.0f;
    double bandwidth_GBps = 0.0;

    // --- Pageable Host to Device ---
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaMemcpy(d_mem, h_pageable, BUF_SIZE, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    bandwidth_GBps = (double)BUF_SIZE * NUM_ITER / (elapsedTime * 1e6);
    double pageable_HtoD_GBps = bandwidth_GBps;

    // --- Pinned Host to Device ---
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaMemcpy(d_mem, h_pinned, BUF_SIZE, cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    bandwidth_GBps = (double)BUF_SIZE * NUM_ITER / (elapsedTime * 1e6);
    double pinned_HtoD_GBps = bandwidth_GBps;

    // --- Pageable Device to Host ---
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaMemcpy(h_pageable, d_mem, BUF_SIZE, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    bandwidth_GBps = (double)BUF_SIZE * NUM_ITER / (elapsedTime * 1e6);
    double pageable_DtoH_GBps = bandwidth_GBps;

    // --- Pinned Device to Host ---
    CHECK_CUDA(cudaEventRecord(start, 0));
    for (int i = 0; i < NUM_ITER; ++i) {
        CHECK_CUDA(cudaMemcpy(h_pinned, d_mem, BUF_SIZE, cudaMemcpyDeviceToHost));
    }
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));
    bandwidth_GBps = (double)BUF_SIZE * NUM_ITER / (elapsedTime * 1e6);
    double pinned_DtoH_GBps = bandwidth_GBps;

    // Print the bandwidth table
    printf("\nMemory Copy Bandwidths (GB/s)\n");
    printf("----------------------------------------------\n");
    printf("| %-12s | %-12s |\n", "Copy Type", "Bandwidth");
    printf("----------------------------------------------\n");
    printf("| %-12s | %10.2f |\n", "Pageable HtoD", pageable_HtoD_GBps);
    printf("| %-12s | %10.2f |\n", "Pinned   HtoD", pinned_HtoD_GBps);
    printf("| %-12s | %10.2f |\n", "Pageable DtoH", pageable_DtoH_GBps);
    printf("| %-12s | %10.2f |\n", "Pinned   DtoH", pinned_DtoH_GBps);
    printf("----------------------------------------------\n\n");

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_mem));
    CHECK_CUDA(cudaFreeHost(h_pinned));
    free(h_pageable);

    return EXIT_SUCCESS;
}
```