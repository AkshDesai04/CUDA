```c
/*
Aim of the program:
Plot the measured bandwidth for different transfer sizes (e.g., 1KB, 1MB, 16MB, 128MB). Notice that bandwidth is lower for small transfers due to latency overhead.

Thinking:
The goal is to empirically demonstrate the effect of transfer size on effective bandwidth between host and device memory in CUDA. Because CUDA memcpy operations incur a fixed latency overhead (e.g., due to driver and PCIe handshake), small payloads suffer a disproportionately low effective bandwidth. As the payload grows, this overhead becomes negligible compared to the transfer time, and the bandwidth approaches the theoretical limit of the underlying interconnect (e.g., PCIe, NVLink).

To showcase this effect we:
1. Define a set of representative transfer sizes: 1 KB, 1 MB, 16 MB, and 128 MB.
2. For each size:
   - Allocate pinned host memory with cudaMallocHost (to avoid extra copies).
   - Allocate device memory with cudaMalloc.
   - Initialize host buffer with arbitrary data.
   - Time a host‑to‑device copy using CUDA events.
   - Time a device‑to‑host copy using CUDA events.
   - Compute effective bandwidth in GB/s for each direction.
3. Output a table of results. The user can redirect the output to a file and plot with any external tool (e.g., gnuplot, matplotlib).
4. Use a simple error‑checking macro for robustness.

The program does not plot directly; it prints results so the user can plot them externally. This keeps the CUDA code portable and focused on the core measurement task.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(err)                                               \
    do {                                                              \
        cudaError_t err__ = (err);                                    \
        if (err__ != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err__));  \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

int main(void) {
    // Transfer sizes in bytes
    size_t sizes[] = {1024,                 // 1 KB
                      1024 * 1024,          // 1 MB
                      16 * 1024 * 1024,     // 16 MB
                      128 * 1024 * 1024};   // 128 MB
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Transfer Size (Bytes)\tHBW (GB/s)\tDBW (GB/s)\n");
    printf("-----------------------------------------------------\n");

    for (int i = 0; i < num_sizes; ++i) {
        size_t bytes = sizes[i];
        size_t num_bytes = bytes;

        // Allocate pinned host memory
        void *h_ptr = NULL;
        CUDA_CHECK(cudaMallocHost(&h_ptr, num_bytes));

        // Allocate device memory
        void *d_ptr = NULL;
        CUDA_CHECK(cudaMalloc(&d_ptr, num_bytes));

        // Initialize host buffer with dummy data
        memset(h_ptr, 42, num_bytes);

        // Events for timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        float elapsed_ms;

        // Host to Device copy
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(d_ptr, h_ptr, num_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double h2d_bandwidth = (double)num_bytes / (elapsed_ms * 1e6); // bytes per second
        h2d_bandwidth /= 1e9; // GB/s

        // Device to Host copy
        CUDA_CHECK(cudaEventRecord(start, 0));
        CUDA_CHECK(cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double d2h_bandwidth = (double)num_bytes / (elapsed_ms * 1e6); // bytes per second
        d2h_bandwidth /= 1e9; // GB/s

        // Print results
        printf("%-20zu\t%.3f\t\t%.3f\n", num_bytes, h2d_bandwidth, d2h_bandwidth);

        // Clean up
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFreeHost(h_ptr));
        CUDA_CHECK(cudaFree(d_ptr));
    }

    return 0;
}
```