```cuda
/* 
Aim of the program:
Create a loop that performs the HtoD copy multiple times and averages the bandwidth to get a more stable measurement.

Thinking:
The goal is to measure the host‑to‑device (HtoD) memory copy bandwidth using CUDA. 
To obtain a reliable estimate, we will perform the copy operation several times (e.g. 10 iterations) and average the bandwidth results. 
Timing will be done with CUDA events for accurate GPU-side measurement. 
The program will:
1. Allocate a sizable buffer on the host (e.g. 128 MB) and on the device.
2. Create CUDA events for start/stop timing.
3. In a loop of N iterations:
   a. Record the start event.
   b. Perform cudaMemcpy(HtoD).
   c. Record the stop event.
   d. Calculate elapsed time and compute the bandwidth for this iteration.
4. After the loop, compute the average bandwidth and print it.
5. Clean up resources.

We also include a simple error‑checking macro to simplify CUDA API error handling. 
The code is self‑contained and can be compiled with `nvcc`:
    nvcc -O2 -arch=sm_60 -o h2d_bandwidth h2d_bandwidth.cu
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

int main() {
    // Parameters
    const size_t bytes = 128 * 1024 * 1024; // 128 MB
    const int iterations = 10;

    // Allocate pinned host memory for faster transfer
    void* h_ptr = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_ptr, bytes));

    // Allocate device memory
    void* d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));

    // Initialize host memory (optional, but ensures data movement)
    memset(h_ptr, 0xA5, bytes);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    double total_bandwidth_gb_per_s = 0.0;

    printf("Performing %d HtoD copies of %zu MB...\n", iterations, bytes / (1024 * 1024));

    for (int i = 0; i < iterations; ++i) {
        // Record start event
        CHECK_CUDA(cudaEventRecord(start, 0));

        // HtoD copy
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));

        // Record stop event
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        // Calculate elapsed time in milliseconds
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

        // Convert to seconds
        double sec = ms / 1000.0;

        // Bandwidth in GB/s
        double bandwidth = (bytes / (1024.0 * 1024.0 * 1024.0)) / sec;

        printf("Iteration %2d: %.3f GB/s\n", i + 1, bandwidth);

        total_bandwidth_gb_per_s += bandwidth;
    }

    double avg_bandwidth = total_bandwidth_gb_per_s / iterations;
    printf("\nAverage HtoD bandwidth: %.3f GB/s\n", avg_bandwidth);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaFreeHost(h_ptr));

    return 0;
}
```