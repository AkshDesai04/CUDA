/*
Aim of the program:
Check and print if the device can overlap a `memcpy` operation with kernel execution (`deviceOverlap`).

Thinking:
- Use the CUDA Runtime API to query the device attribute `cudaDevAttrDeviceOverlap`.
- The attribute returns 0 or 1 indicating whether the device can concurrently execute kernels and memory copies.
- Retrieve the attribute for the current device (device 0 by default).
- Print a human‑readable message indicating support.
- Handle any CUDA API errors gracefully by printing an error message and exiting.
- The program is written in C/CUDA, compiled with `nvcc` to produce a .cu file.
- No external dependencies or supporting files are required – the code is self‑contained.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro to check CUDA API return codes
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                  \
        }                                                          \
    } while (0)

int main(void) {
    int device = 0;  // Use the default device (device 0)
    int canOverlap = 0;

    // Query the device attribute cudaDevAttrDeviceOverlap
    CHECK_CUDA(cudaDeviceGetAttribute(&canOverlap,
                                      cudaDevAttrDeviceOverlap,
                                      device));

    if (canOverlap) {
        printf("Device %d can overlap memcpy and kernel execution: YES\n", device);
    } else {
        printf("Device %d can overlap memcpy and kernel execution: NO\n", device);
    }

    return EXIT_SUCCESS;
}
