/*
Time the duration of the empty kernel launch using `cudaEvent` timers.

Thinking:
The goal is to illustrate how to measure GPU kernel launch time using CUDA's event-based timers.  
We create two `cudaEvent_t` objects, one to mark the start and one to mark the end of the empty kernel launch.  
We record the start event, launch an empty kernel (one that does nothing but is still a valid kernel launch), record the stop event, and then synchronize on the stop event to ensure the kernel has finished executing.  
After synchronization we compute the elapsed time in milliseconds with `cudaEventElapsedTime` and print the result.  
This example also includes basic error checking to make sure the CUDA API calls succeed.  
Since the kernel is empty, the measured time reflects primarily the overhead of launching a kernel on the GPU, which is useful for performance studies or for comparing different launch configurations.
*/

#include <iostream>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__   \
                      << " - " << cudaGetErrorString(err) << std::endl;    \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Empty kernel that does nothing
__global__ void emptyKernel()
{
    // No operation
}

int main()
{
    // Create CUDA events for timing
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    // Record the start event
    CUDA_CHECK(cudaEventRecord(startEvent, 0));

    // Launch the empty kernel
    emptyKernel<<<1, 1>>>();

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stopEvent, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    // Calculate elapsed time in milliseconds
    float elapsedTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTimeMs, startEvent, stopEvent));

    // Output the measured time
    std::cout << "Empty kernel launch time: " << elapsedTimeMs << " ms" << std::endl;

    // Clean up events
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    return 0;
}
