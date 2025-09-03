/*
Write a program that deliberately triggers a `cudaErrorInvalidDevice` by trying to `cudaSetDevice()` to a device index that doesn't exist (e.g., 99). Use your `CHECK()` macro to catch it.

Thinking:
- The goal is to show how a CUDA error can be caught with a macro.
- The macro will wrap a CUDA runtime call, check the returned cudaError_t, and if it's not cudaSuccess, print the error string and exit.
- The program will simply attempt to set an invalid device number (99), which is unlikely to exist on any system.
- We will include the necessary headers and provide a minimal main function.
- After attempting to set the device, the program will exit. There's no need for device memory allocation or kernel launches.
- This example demonstrates error handling and the use of the CHECK macro.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro to check CUDA errors
#define CHECK(call)                                            \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(void) {
    // Intentionally use an invalid device index
    int invalid_device = 99;
    CHECK(cudaSetDevice(invalid_device));

    // If we get here, something unexpected happened
    printf("Unexpectedly succeeded in setting device %d\n", invalid_device);
    return EXIT_SUCCESS;
}
