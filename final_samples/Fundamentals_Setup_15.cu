```cpp
/*
Write a program that resets the primary CUDA device using `cudaDeviceReset()`. Explain in a comment why and when this function might be useful (e.g., for cleaning up resources or profiling).

Thinking:
- The purpose of this program is to demonstrate the simplest use case of `cudaDeviceReset()`, which tells the CUDA runtime to release all resources (memory, streams, events, contexts, etc.) associated with the current device and reset it to its initial state.
- This is often used at the end of a CUDA application to ensure a clean shutdown, especially when the program might be invoked multiple times in a long-running host process or a unit test suite.
- `cudaDeviceReset()` is also helpful for profiling or debugging: tools like Nsight Systems or Nsight Compute can start with a clean device state, and any artifacts from previous runs are cleared.
- Additionally, it is useful for testing device context recreation, and it guarantees that any lingering device-side state (e.g., allocated memory) does not affect subsequent code.
- We will check for errors from `cudaDeviceReset()` and print a message indicating success or failure. The program will be minimal, focusing solely on the reset operation.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Utility macro for error checking
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                               \
        }                                                      \
    } while (0)

int main(void) {
    // Reset the primary CUDA device
    CHECK_CUDA(cudaDeviceReset());

    printf("Device reset successfully.\n");
    return EXIT_SUCCESS;
}
```