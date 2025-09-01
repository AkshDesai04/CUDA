/*
Create a loop that performs the HtoD copy multiple times and averages the bandwidth to get a more stable measurement.

[Thinking]  
The goal is to implement a small CUDA benchmark that measures the bandwidth of host‑to‑device memory copies.  
A single copy can produce noisy results due to kernel launch overhead, cache effects, and background system activity.  
To mitigate this, we will:

1. Allocate a large buffer on the host and the device (e.g., 64 MB) to ensure the copy time dominates over other overheads.
2. Use CUDA events (`cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`) to capture the time of each individual `cudaMemcpy` call.
3. Execute the copy in a loop (e.g., 100 iterations), summing the elapsed times.
4. Compute the bandwidth for each iteration as `(size in bytes) / (time in seconds)` and accumulate it.
5. After the loop, calculate the average bandwidth across all iterations and report it in GB/s.
6. Include basic error checking via a macro to keep the code readable.
7. Use C99 syntax for the host code but compile with `nvcc` as a CUDA C source file (.cu).
8. Keep the code minimal and self‑contained, so it can be compiled and run on any system with an appropriate NVIDIA GPU.

The program will print:
```
Total iterations: N
Average HtoD bandwidth: X.XX GB/s
```
and exit cleanly.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void)
{
    /* Configuration */
    const size_t MB = 64;                    /* Size per buffer in megabytes */
    const size_t bytes = MB * 1024ULL * 1024ULL;
    const int iterations = 100;              /* Number of HtoD copies */

    /* Allocate pinned host memory for better transfer speed */
    float *h_buf = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_buf, bytes));

    /* Allocate device memory */
    float *d_buf = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_buf, bytes));

    /* Initialize host buffer with dummy data */
    for (size_t i = 0; i < bytes / sizeof(float); ++i) {
        h_buf[i] = (float)i;
    }

    /* Create CUDA events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_time_ms = 0.0;  /* Accumulated time in milliseconds */
    double total_bandwidth = 0.0;/* Accumulated bandwidth in GB/s */

    for (int i = 0; i < iterations; ++i) {
        CUDA_CHECK(cudaEventRecord(start, 0));

        /* Perform host-to-device copy */
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

        double elapsed_s = elapsed_ms / 1000.0;
        double bandwidth = (bytes / 1e9) / elapsed_s; /* GB/s */

        total_time_ms += elapsed_ms;
        total_bandwidth += bandwidth;
    }

    double avg_bandwidth = total_bandwidth / iterations;

    printf("Total iterations: %d\n", iterations);
    printf("Average HtoD bandwidth: %.2f GB/s\n", avg_bandwidth);

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFreeHost(h_buf));

    return 0;
}
