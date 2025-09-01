/*
Aim of the program:
Write a program that queries the available global memory and prints a warning if it is less than a certain threshold (e.g., 4096 MB).

Thinking:
- The program will use the CUDA runtime API to obtain memory information from the GPU.
- First, we query the number of CUDA devices available with `cudaGetDeviceCount`. If no devices are found, we print an error and exit.
- We set the current device to the first GPU (`cudaSetDevice(0)`).
- The `cudaMemGetInfo` function returns the amount of free and total device memory in bytes for the current device. This is the "available global memory" the prompt refers to.
- We convert a threshold value expressed in megabytes (4096 MB) to bytes by multiplying by `1024 * 1024`.
- If the free memory is less than the threshold, we print a warning message. Otherwise, we print that the memory is sufficient.
- For better readability, we also output the free and total memory in megabytes.
- Error checking is performed after each CUDA API call; a simple macro `CHECK_CUDA_ERROR` is used to handle failures gracefully.
- The program compiles with `nvcc` and requires no device code; it is purely host code.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro to check CUDA errors
#define CHECK_CUDA_ERROR(call)                                               \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device
    CHECK_CUDA_ERROR(cudaSetDevice(0));

    size_t freeMemBytes = 0;
    size_t totalMemBytes = 0;
    CHECK_CUDA_ERROR(cudaMemGetInfo(&freeMemBytes, &totalMemBytes));

    const size_t thresholdBytes = 4096ULL * 1024ULL * 1024ULL; // 4096 MB in bytes

    printf("Device 0 memory information:\n");
    printf("  Total memory:   %zu MB\n", totalMemBytes / (1024 * 1024));
    printf("  Free memory:    %zu MB\n", freeMemBytes / (1024 * 1024));

    if (freeMemBytes < thresholdBytes) {
        fprintf(stderr,
                "WARNING: Available global memory (%zu MB) is below the "
                "threshold of %zu MB.\n",
                freeMemBytes / (1024 * 1024), thresholdBytes / (1024 * 1024));
    } else {
        printf("Available global memory is above the threshold.\n");
    }

    return EXIT_SUCCESS;
}
