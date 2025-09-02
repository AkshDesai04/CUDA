```cuda
/*
Create a loop that performs the HtoD copy multiple times and averages the bandwidth to get a more stable measurement.

Thinking:
- The goal is to benchmark host-to-device (HtoD) memory copy performance.
- We will allocate a large buffer on the host and device (e.g., 100 MB).
- Use cudaMemcpy with cudaMemcpyHostToDevice to perform the copy.
- To get a stable measurement, repeat the copy many times (e.g., 100 iterations) and accumulate elapsed time.
- Use cudaEvent_t for timing because it measures device-related operations accurately.
- After the loop, compute average bandwidth: total_bytes / total_time.
- Include error checking after each CUDA API call.
- Use a simple main function that prints the average bandwidth in GB/s.
- Ensure code compiles with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main(void) {
    const size_t bytes = 100 * 1024 * 1024; // 100 MB
    const int iterations = 100;

    // Allocate host memory
    void *h_ptr = NULL;
    CHECK_CUDA(cudaMallocHost(&h_ptr, bytes)); // page-locked for faster transfer

    // Allocate device memory
    void *d_ptr = NULL;
    CHECK_CUDA(cudaMalloc(&d_ptr, bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;

    for (int i = 0; i < iterations; ++i) {
        CHECK_CUDA(cudaEventRecord(start, 0));
        CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop, 0));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
    }

    // Compute average bandwidth in GB/s
    double avg_ms = total_ms / iterations;
    double bandwidth_gb_s = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg_ms / 1000.0);

    printf("Average HtoD bandwidth: %.2f GB/s over %d iterations (%.3f ms each)\n",
           bandwidth_gb_s, iterations, avg_ms);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_ptr));
    CHECK_CUDA(cudaFreeHost(h_ptr));

    return 0;
}
```