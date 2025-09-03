/*
Write a program that resets the primary CUDA device using `cudaDeviceReset()`. Explain in a comment why and when this function might be useful (e.g., for cleaning up resources or profiling).

Thought process:
- The objective is to create a minimal CUDA host program that demonstrates the use of cudaDeviceReset().
- The program should include necessary headers (<cuda_runtime.h>) and a main function.
- Inside main, we can perform a trivial CUDA call such as cudaGetDevice() to ensure the device is initialized.
- Then call cudaDeviceReset() to reset the device state.
- We need to check the return status of each CUDA API call and print errors if any.
- After resetting, we can optionally query the device again to confirm reset behavior.
- Provide a comment explaining that cudaDeviceReset() cleans up all allocated resources on the device, resets device state, and is useful at program exit for profiling or ensuring a clean slate for subsequent runs.
- Ensure that the program is self-contained and compiles as a .cu file.

*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void) {
    // Get current device to ensure the device is initialized
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    printf("Current device: %d\n", device);

    // Optionally, perform a trivial kernel launch or memory operation
    // Here we just allocate and free a small piece of device memory
    int *d_ptr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_ptr, 4 * sizeof(int)));
    CHECK_CUDA(cudaFree(d_ptr));

    // Reset the primary CUDA device
    // This function cleans up all resources allocated by the program
    // on the GPU (memory, contexts, etc.) and resets the device state.
    // It is especially useful:
    //   * when profiling or collecting metrics, to ensure no leftover state
    //     interferes with subsequent measurements.
    //   * at program termination to release resources in a deterministic way.
    //   * when debugging, to reset the device to a known clean state.
    printf("Resetting device...\n");
    CHECK_CUDA(cudaDeviceReset());
    printf("Device reset successful.\n");

    // After reset, we can query the device again to confirm state
    int device_after_reset;
    CHECK_CUDA(cudaGetDevice(&device_after_reset));
    printf("Device after reset: %d\n", device_after_reset);

    return 0;
}
