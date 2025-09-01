```cuda
/*
The formula for bandwidth is `(bytes / 1e9) / (milliseconds / 1000.0)`.

My thinking:
I will write a simple CUDA program that allocates a large array on the host, copies it to the device, then copies it back to the host. I will use CUDA events to time the combined copy operation (host→device + device→host). The total number of bytes transferred is the size of the array times two (since we transfer it twice). After the transfer completes I will compute the bandwidth in GB/s using the formula provided:
    bandwidth = (bytes / 1e9) / (milliseconds / 1000.0)
The program will print the bandwidth to standard output. I will include basic error checking for CUDA API calls and keep the code minimal and self‑contained in a single .cu file. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            return EXIT_FAILURE;                                       \
        }                                                               \
    } while (0)

int main(void) {
    const size_t N = 100 * 1024 * 1024;          // 100 million elements
    const size_t SIZE = N * sizeof(float);       // bytes

    // Allocate host memory
    float *h_data = (float *)malloc(SIZE);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_data, SIZE));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Host to Device copy
    CUDA_CHECK(cudaMemcpy(d_data, h_data, SIZE, cudaMemcpyHostToDevice));

    // Device to Host copy
    CUDA_CHECK(cudaMemcpy(h_data, d_data, SIZE, cudaMemcpyDeviceToHost));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Total bytes transferred: host→device + device→host
    double total_bytes = 2.0 * SIZE;  // bytes

    // Compute bandwidth in GB/s using the provided formula
    double bandwidth = (total_bytes / 1e9) / (ms / 1000.0);

    printf("Data size: %.2f MB\n", total_bytes / (1024.0 * 1024.0));
    printf("Elapsed time: %.3f ms\n", ms);
    printf("Bandwidth: %.3f GB/s\n", bandwidth);

    // Clean up memory
    free(h_data);
    CUDA_CHECK(cudaFree(d_data));

    return EXIT_SUCCESS;
}
```