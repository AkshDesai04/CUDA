```c
/*
Time the data transfer speed (bandwidth) of `cudaMemcpy`. Transfer a large array (e.g., 256MB) from host to device. Record the time before and after the copy using a CUDA event timer (`cudaEventCreate`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaEventElapsedTime`). Calculate and print the bandwidth in GB/s.

The program aims to benchmark the host-to-device transfer throughput by allocating a sizable array on the CPU, copying it to the GPU, and measuring the elapsed time using CUDA events. It then computes the bandwidth in gigabytes per second (GiB/s) by dividing the total data size by the transfer time. This gives a clear performance metric for the system's memory bus. The code follows these steps:
1. Define a 256‑MB data size.
2. Allocate host memory (page‑locked) for efficient transfer.
3. Allocate device memory.
4. Create CUDA events for timing.
5. Record start event, perform `cudaMemcpy`, record end event.
6. Synchronize, compute elapsed time, calculate bandwidth, and print.
7. Clean up resources.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    const size_t MB = 1024 * 1024;
    const size_t numBytes = 256 * MB;  // 256 MB
    const size_t numElements = numBytes / sizeof(float);

    // Allocate pinned host memory for faster transfer
    float *h_data = NULL;
    cudaError_t err = cudaMallocHost((void**)&h_data, numBytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned host memory (error code %d)!\n", err);
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < numElements; ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_data = NULL;
    err = cudaMalloc((void**)&d_data, numBytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %d)!\n", err);
        cudaFreeHost(h_data);
        return EXIT_FAILURE;
    }

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, 0);

    // Perform host-to-device copy
    err = cudaMemcpy(d_data, h_data, numBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory to device (error code %d)!\n", err);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_data);
        cudaFreeHost(h_data);
        return EXIT_FAILURE;
    }

    // Record the stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Compute bandwidth in GiB/s (Gibibytes per second)
    double elapsedSeconds = milliseconds / 1000.0;
    double bandwidthGiBps = (double)numBytes / (1024.0 * 1024.0 * 1024.0) / elapsedSeconds;

    // Print results
    printf("Transferred %zu MB in %f ms.\n", numBytes / MB, milliseconds);
    printf("Bandwidth: %f GiB/s.\n", bandwidthGiBps);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFreeHost(h_data);

    return EXIT_SUCCESS;
}
```