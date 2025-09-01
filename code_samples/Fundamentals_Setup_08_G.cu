/*
 * Aim of the program: Check if the device is a multi-GPU board (`isMultiGpuBoard`).
 *
 * Thinking process:
 * 1. Determine what the user wants: a CUDA program that checks the `isMultiGpuBoard` flag for each
 *    device available on the system. The `cudaDeviceProp` structure contains a field named
 *    `multiGpuBoard` (available from CUDA 7.0 onwards). If this field is non‑zero, the device
 *    belongs to a multi‑GPU board.
 *
 * 2. The program will:
 *    - Include the CUDA runtime header `<cuda_runtime.h>`.
 *    - Use `cudaGetDeviceCount` to obtain the number of GPUs.
 *    - Iterate over each device id, retrieving its properties with `cudaGetDeviceProperties`.
 *    - Print the device name and whether `multiGpuBoard` is true or false.
 *
 * 3. Error handling: Wrap CUDA API calls in a helper macro that checks the returned status and
 *    prints an error message before exiting if anything goes wrong.
 *
 * 4. Keep the code pure C with minimal dependencies: use `printf` for output, no C++ I/O.
 *
 * 5. The resulting file is a complete `.cu` source file ready for compilation with `nvcc`.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors following a CUDA API call. */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Detected %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Multi-GPU board: %s\n", prop.multiGpuBoard ? "YES" : "NO");
        printf("\n");
    }

    return 0;
}
