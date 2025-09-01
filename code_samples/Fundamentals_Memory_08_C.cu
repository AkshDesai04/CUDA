/*
Create a loop that performs the HtoD copy multiple times and averages the bandwidth to get a more stable measurement.

Thought process:
- The core requirement is to repeatedly copy data from host to device and measure the bandwidth.
- To obtain a stable measurement, the code will perform the copy a fixed number of times (e.g., 10) and average the bandwidth across those runs.
- We'll use CUDA events (cudaEvent_t) for timing because they provide high-resolution timing suitable for measuring data transfer times.
- The program will allocate a buffer on the host, fill it with some data, allocate a corresponding buffer on the device, then perform the copy in a loop.
- After each copy, the elapsed time will be recorded. The bandwidth for that iteration is calculated as (size in bytes) / (elapsed time in seconds). Summing these values and dividing by the number of iterations yields the average bandwidth.
- The code includes basic error checking through a macro that verifies CUDA API calls.
- After the loop, the program will print the average bandwidth in GB/s.
- Finally, resources (memory, events) are cleaned up.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    const size_t size = 256 * 1024 * 1024; // 256 MB
    const int numRuns = 10;

    // Allocate pinned host memory for faster transfer
    float *h_data = NULL;
    CUDA_CHECK(cudaMallocHost((void**)&h_data, size));

    // Fill host memory with some values
    for (size_t i = 0; i < size / sizeof(float); ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, size));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float totalBandwidthGBs = 0.0f;

    for (int run = 0; run < numRuns; ++run) {
        CUDA_CHECK(cudaEventRecord(start, 0));

        // Perform host-to-device copy
        CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        // Compute bandwidth: size (bytes) / time (seconds)
        float seconds = milliseconds / 1000.0f;
        float bandwidthGBs = (size / (1024.0f * 1024.0f * 1024.0f)) / seconds;
        totalBandwidthGBs += bandwidthGBs;

        printf("Run %2d: %8.3f GB/s\n", run + 1, bandwidthGBs);
    }

    float avgBandwidthGBs = totalBandwidthGBs / numRuns;
    printf("\nAverage bandwidth over %d runs: %8.3f GB/s\n", numRuns, avgBandwidthGBs);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFreeHost(h_data));

    return 0;
}
