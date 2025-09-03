/*
Aim: Time the duration of the empty kernel launch using cudaEvent timers.

Thinking:
1. Create an empty CUDA kernel that performs no operations.
2. Use CUDA events (cudaEventCreate) to capture timestamps before and after the kernel launch.
3. Record a start event, launch the kernel asynchronously, record a stop event.
4. Synchronize on the stop event to ensure the kernel has finished executing.
5. Compute the elapsed time with cudaEventElapsedTime, which returns time in milliseconds.
6. Print the measured duration.
7. Add simple error checking macro to catch CUDA API errors.
8. Clean up events and return from main.

This minimal program demonstrates how to use CUDA events to time the duration of an empty kernel launch. The measured time will be dominated by overhead such as kernel launch latency and event recording overhead, giving insight into the baseline overhead of kernel execution on the device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Empty kernel: does nothing
__global__ void emptyKernel()
{
    // No work performed
}

int main(void)
{
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch empty kernel (1 block, 1 thread)
    emptyKernel<<<1, 1>>>();

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete (ensures kernel finished)
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time in milliseconds
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    // Print the result
    printf("Elapsed time for empty kernel launch: %f ms\n", ms);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Reset device (optional but good practice)
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}
