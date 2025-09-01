```cuda
/*
Aim of the program:
Plot the measured bandwidth for different transfer sizes (e.g., 1KB, 1MB, 16MB, 128MB). Notice that bandwidth is lower for small transfers due to latency overhead.

Thinking:
To demonstrate the impact of latency on transfer bandwidth, the program will perform host-to-device and device-to-host copies for several block sizes: 1KB, 1MB, 16MB, and 128MB.  
CUDA events are used to accurately time each transfer because they provide nanosecond resolution and are device‑synchronized.  
Bandwidth is computed as (size in bytes) / (elapsed time in seconds). The program will print a table of sizes and corresponding bandwidths for both copy directions.  
A real plot can be generated from this output using any external plotting tool (e.g., gnuplot, Excel, matplotlib).  
The program is kept minimal and self‑contained; it does not rely on any external libraries beyond the CUDA runtime.  
Edge cases such as sizes exceeding device memory are handled by checking the allocation result.  

Implementation details:
1. Define the transfer sizes in an array of unsigned integers (bytes).  
2. For each size:
   - Allocate host memory (pinned) using cudaMallocHost for better transfer performance.  
   - Allocate device memory using cudaMalloc.  
   - Warm up the copy once before timing to avoid first‑time overhead.  
   - Create CUDA events for start and stop.  
   - Measure H2D and D2H copy times.  
   - Compute bandwidth in GB/s.  
   - Print results.  
3. Free all allocated memory and destroy events.  
4. The program exits with status 0 on success.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    // Transfer sizes in bytes
    const size_t sizes[] = {
        1 * 1024,          // 1KB
        1 * 1024 * 1024,   // 1MB
        16 * 1024 * 1024,  // 16MB
        128 * 1024 * 1024  // 128MB
    };
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Transfer Size\tH2D Bandwidth (GB/s)\tD2H Bandwidth (GB/s)\n");
    printf("----------------------------------------------------------------\n");

    for (int i = 0; i < num_sizes; ++i) {
        size_t size = sizes[i];

        // Allocate pinned host memory
        void *h_ptr = nullptr;
        CHECK_CUDA(cudaMallocHost(&h_ptr, size));

        // Allocate device memory
        void *d_ptr = nullptr;
        cudaError_t err = cudaMalloc(&d_ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate %zu bytes on device: %s\n",
                    size, cudaGetErrorString(err));
            cudaFreeHost(h_ptr);
            continue; // skip this size
        }

        // Warm up copy
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));

        // Events for timing
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        // Host to Device timing
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float h2d_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start, stop));

        // Device to Host timing
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, size, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float d2h_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start, stop));

        // Compute bandwidth in GB/s
        double h2d_bw = (double)size / (h2d_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        double d2h_bw = (double)size / (d2h_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);

        printf("%8zu bytes\t%12.2f\t\t%12.2f\n", size, h2d_bw, d2h_bw);

        // Clean up
        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
        CHECK_CUDA(cudaFree(d_ptr));
        CHECK_CUDA(cudaFreeHost(h_ptr));
    }

    return 0;
}
```