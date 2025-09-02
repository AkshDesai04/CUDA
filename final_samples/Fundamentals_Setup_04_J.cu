/*
Query and print whether the device can map host memory (`canMapHostMemory` property).

Thinking process:
- The goal is to write a simple CUDA program in C that determines whether the current CUDA device supports mapping host memory into the device's address space.
- In CUDA, the device capabilities are exposed through the `cudaDeviceProp` structure, which is filled by `cudaGetDeviceProperties()`.
- The `canMapHostMemory` field of this structure indicates whether the device can map host memory.
- The program should:
  1. Determine the number of available CUDA devices.
  2. Choose a device (device 0 for simplicity).
  3. Retrieve its properties.
  4. Print the value of `canMapHostMemory` as a human‑readable "yes" or "no".
- Error checking will be performed for CUDA API calls using a simple macro that prints an error message and exits if a call fails.
- The code will be self‑contained, compile with `nvcc`, and produce a console output indicating the mapping capability.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA API calls
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                      \
        }                                                             \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use device 0
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device \"%s\" (ID %d) can map host memory: %s\n",
           prop.name, device,
           prop.canMapHostMemory ? "yes" : "no");

    return EXIT_SUCCESS;
}
