```cpp
/*
Aim of the program: Time the data transfer speed (bandwidth) of cudaMemcpy. Transfer a large array (e.g., 256MB) from host to device. Record the time before and after the copy using a CUDA event timer (cudaEventCreate, cudaEventRecord, cudaEventSynchronize, cudaEventElapsedTime). Calculate and print the bandwidth in GB/s.

Thinking:
1. Choose a data type and size that gives exactly 256MB. Using float (4 bytes) we need 256MB / 4 = 64 million elements.
2. Allocate host memory with malloc or new. Initialize to some data (not necessary for bandwidth measurement).
3. Allocate device memory with cudaMalloc.
4. Create CUDA events for start and stop.
5. Record start event, perform cudaMemcpy from host to device, record stop event.
6. Synchronize on stop event to ensure copy finished.
7. Use cudaEventElapsedTime to get time in milliseconds.
8. Compute bandwidth: bytes transferred / (time_ms / 1000) = bytes per second. Convert to GB/s by dividing by (1024^3).
9. Print the result.
10. Clean up: free host memory, free device memory, destroy events.
11. Add minimal error checking after CUDA calls for robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    // Size of the array: 256 MB
    const size_t bytes = 256 * 1024 * 1024;            // 256 MB in bytes
    const size_t numElements = bytes / sizeof(float); // Number of float elements

    // Allocate host memory
    float* h_data = (float*)malloc(bytes);
    if (h_data == nullptr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }

    // Initialize host data (optional for bandwidth test)
    for (size_t i = 0; i < numElements; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float* d_data = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        free(h_data);
        return -1;
    }

    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Record start event
    cudaEventRecord(startEvent, 0);

    // Perform the copy: host to device
    err = cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        // Cleanup
        cudaFree(d_data);
        free(h_data);
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        return -1;
    }

    // Record stop event
    cudaEventRecord(stopEvent, 0);

    // Wait for the stop event to complete
    cudaEventSynchronize(stopEvent);

    // Compute elapsed time in milliseconds
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, startEvent, stopEvent);

    // Calculate bandwidth in GB/s
    // elapsed_ms is in milliseconds, convert to seconds
    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = (double)bytes / elapsed_sec / (1024.0 * 1024.0 * 1024.0);

    // Print the result
    printf("Transfered %zu bytes in %.3f ms\n", bytes, elapsed_ms);
    printf("Bandwidth: %.3f GB/s\n", bandwidth_GBps);

    // Cleanup
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
```