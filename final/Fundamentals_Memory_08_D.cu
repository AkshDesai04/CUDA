```cpp
/*
Aim of the program:
Plot the measured bandwidth for different transfer sizes (e.g., 1KB, 1MB, 16MB, 128MB). Notice that bandwidth is lower for small transfers due to latency overhead.

Thinking and design:
- The goal is to empirically measure the effective bandwidth of PCI‑e transfers between the host and the device for several sizes.
- To avoid host side memory allocation overhead skewing the results, we use pinned (page‑locked) memory via cudaMallocHost.  
- For accurate timing we use CUDA events (cudaEvent_t) because they measure GPU‑side activity and are more precise than CPU timers for short bursts.
- We perform a round‑trip copy (H2D + D2H) and average over several iterations (e.g., 10) to reduce noise.  
- Bandwidth is computed as total bytes transferred divided by the elapsed time, expressed in GiB/s.
- Results are printed to stdout. The user can redirect the output to a file and plot with an external tool (e.g., gnuplot, matplotlib).
- The program includes a simple macro for error checking (`CHECK_CUDA`).  
- Memory for each transfer size is allocated once; we reuse it across iterations.  
- Since the largest test is 128 MiB, the program expects the GPU to have at least that much free memory; we guard against allocation failure.

Implementation notes:
- Transfer sizes: 1 KiB, 1 MiB, 16 MiB, 128 MiB (1<<10, 1<<20, 16<<20, 128<<20).
- We perform `N_ITER` iterations (10) per size.
- Bandwidth calculation: bytes * N_ITER * 2 (both directions) / time_in_seconds.
- We print the size, time per iteration, and bandwidth.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void) {
    const int N_ITER = 10;
    const size_t sizes[] = {1024,           // 1 KiB
                            1 << 20,        // 1 MiB
                            16 << 20,       // 16 MiB
                            128 << 20};     // 128 MiB
    const int NUM_SIZES = sizeof(sizes) / sizeof(sizes[0]);

    for (int idx = 0; idx < NUM_SIZES; ++idx) {
        size_t bytes = sizes[idx];
        size_t alloc_bytes = bytes;

        // Allocate pinned host memory
        void* h_ptr = NULL;
        CHECK_CUDA(cudaMallocHost(&h_ptr, alloc_bytes));

        // Allocate device memory
        void* d_ptr = NULL;
        CHECK_CUDA(cudaMalloc(&d_ptr, alloc_bytes));

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Warm‑up copy to mitigate first‑time overhead
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));

        // Timing loop
        float total_ms = 0.0f;
        for (int iter = 0; iter < N_ITER; ++iter) {
            CHECK_CUDA(cudaEventRecord(start, 0));

            // Host to Device copy
            CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));

            // Device to Host copy
            CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost));

            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
            total_ms += ms;
        }

        float avg_ms = total_ms / N_ITER;
        float time_sec = avg_ms / 1000.0f;
        // Total bytes transferred = 2 * bytes * N_ITER (H2D + D2H)
        double bandwidth = (double)(2 * bytes * N_ITER) / time_sec / (1024.0 * 1024.0 * 1024.0); // GiB/s

        printf("Transfer size: %6zu bytes | Avg time: %8.3f ms | Bandwidth: %7.3f GiB/s\n",
               bytes, avg_ms, bandwidth);

        // Cleanup
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_ptr));
        CHECK_CUDA(cudaFreeHost(h_ptr));
    }

    return 0;
}
```