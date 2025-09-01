```cpp
/*
 * Time the duration of the empty kernel launch using `cudaEvent` timers.
 *
 * Thinking:
 *   1. Use CUDA runtime API to create two events: start and stop.
 *   2. Record the start event just before launching the kernel.
 *   3. Launch a kernel that does nothing. This kernel is defined as
 *      __global__ void emptyKernel() { }.
 *   4. Record the stop event immediately after the kernel launch. CUDA
 *      guarantees that the event will be recorded after the kernel
 *      completes, but to be safe we call cudaEventSynchronize on the stop
 *      event before measuring elapsed time.
 *   5. Use cudaEventElapsedTime to get the elapsed time in milliseconds
 *      and print it.
 *   6. Add basic error checking via a macro to ensure that any CUDA API
 *      error is reported and the program exits.
 *
 * The program is minimal, compiled with nvcc, and demonstrates the
 * overhead of launching an empty kernel measured with high-resolution
 * CUDA events.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Empty kernel that performs no operations
__global__ void emptyKernel() {
    // Intentionally left blank
}

int main(void) {
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch the empty kernel with a single block of one thread
    emptyKernel<<<1, 1>>>();

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete to ensure kernel has finished
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Print the elapsed time
    printf("Empty kernel launch time: %f ms\n", milliseconds);

    // Clean up events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```