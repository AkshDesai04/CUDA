/*
Use the timer to time a `cudaMemcpy` operation.

The aim of the program is to demonstrate how to measure the time taken for a CUDA memory copy operation (`cudaMemcpy`) using CUDA's event-based timing mechanism. This is useful for performance analysis of GPU kernels and data transfer bottlenecks. The program performs the following steps:

1. Allocate a large array on the host (CPU) and initialize it with some data.
2. Allocate a corresponding array on the device (GPU) memory.
3. Create CUDA events to record timestamps before and after the `cudaMemcpy` operation.
4. Record the start event, perform `cudaMemcpy` from host to device, record the stop event.
5. Synchronize the stop event to ensure the copy has completed.
6. Calculate the elapsed time in milliseconds using `cudaEventElapsedTime`.
7. Print the elapsed time to the console.
8. Clean up all allocated memory and CUDA events.

This program also includes error checking after each CUDA API call. The chosen array size (e.g., 100 million floats) ensures a noticeable transfer time for demonstration purposes, but it can be adjusted as needed.

Thinking process:
- I decided to use CUDA events (`cudaEvent_t`) because they provide high-resolution timing for GPU operations.
- I opted for a simple `float` array because it is easy to allocate and initialize, and large enough to show a meaningful transfer time.
- I wrapped all CUDA API calls in error-checking macros for brevity and safety.
- I used `cudaMemcpyHostToDevice` as the copy direction; `cudaMemcpyDeviceToHost` could be used similarly.
- The program prints both the copy size in bytes and the elapsed time in milliseconds for clarity.
- I added `cudaDeviceReset()` at the end to clean up device state before program exit.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",\
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    }

int main(void) {
    const size_t N = 100 * 1000 * 1000; // 100 million elements
    const size_t size_bytes = N * sizeof(float);

    // Allocate host memory
    float *h_ptr = (float*)malloc(size_bytes);
    if (!h_ptr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (size_t i = 0; i < N; ++i) {
        h_ptr[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_ptr, size_bytes));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Perform the memory copy from host to device
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, size_bytes, cudaMemcpyHostToDevice));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print the timing result
    printf("Transferred %zu bytes from host to device in %f ms.\n",
           size_bytes, milliseconds);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_ptr));
    free(h_ptr);

    // Reset device (optional, but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return EXIT_SUCCESS;
}
