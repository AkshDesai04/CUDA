```cuda
/* 
   Aim of the program: 
   Time the data transfer speed (bandwidth) of `cudaMemcpy`. 
   Transfer a large array (e.g., 256MB) from host to device. 
   Record the time before and after the copy using a CUDA event timer 
   (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`). 
   Calculate and print the bandwidth in GB/s.

   Thought process:
   1. Define the array size: 256 MB = 256 * 1024 * 1024 bytes.
   2. Allocate host memory with `malloc` and device memory with `cudaMalloc`.
   3. Initialize the host array to some pattern (not strictly necessary for timing).
   4. Create CUDA events for start and stop timing.
   5. Record the start event, perform `cudaMemcpy` from host to device,
      then record the stop event and synchronize.
   6. Use `cudaEventElapsedTime` to get elapsed time in milliseconds.
   7. Convert elapsed time to seconds and compute bandwidth:
      bandwidth (GB/s) = (size in bytes) / (elapsed time in seconds) / 1e9.
   8. Print the result with high precision.
   9. Clean up: free host memory, device memory, destroy events.
   10. Include basic error checking after each CUDA API call for robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    const size_t sizeInBytes = 256 * 1024 * 1024; // 256 MB
    const size_t numElements = sizeInBytes / sizeof(float);

    // Allocate host memory
    float *h_array = (float *)malloc(sizeInBytes);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array (optional for timing, but useful for realism)
    for (size_t i = 0; i < numElements; ++i) {
        h_array[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_array = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_array, sizeInBytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Perform the copy from host to device
    CUDA_CHECK(cudaMemcpy(d_array, h_array, sizeInBytes, cudaMemcpyHostToDevice));

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the copy to finish
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    // Compute bandwidth: bytes per second, then convert to GB/s
    double elapsedSec = ms / 1000.0;
    double bandwidthGBps = (static_cast<double>(sizeInBytes) / elapsedSec) / 1e9;

    // Print results
    printf("Copied %zu MB in %f ms\n", sizeInBytes / (1024 * 1024), ms);
    printf("Bandwidth: %f GB/s\n", bandwidthGBps);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_array));
    free(h_array);

    return EXIT_SUCCESS;
}
```