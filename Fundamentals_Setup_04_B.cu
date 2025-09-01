/*
Aim of the program:
Query and print the multiprocessor count for the default device.

Thinking:
1. Include the standard headers needed for CUDA runtime API and I/O.
2. Use `cudaGetDeviceCount` to ensure there is at least one device available.
3. Select the default device (device 0) by using `cudaGetDevice` if necessary, though querying device properties for device 0 suffices.
4. Retrieve device properties using `cudaGetDeviceProperties`.
5. Extract the `multiProcessorCount` field which holds the number of multiprocessors.
6. Print this value to stdout.
7. Handle errors by checking the return status of CUDA API calls, printing informative messages, and exiting with a non-zero status on failure.
8. The program will compile with `nvcc` and run on systems with NVIDIA GPUs.

This straightforward approach demonstrates how to query device capabilities via the CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Utility macro to check CUDA errors */
#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            return EXIT_FAILURE;                                        \
        }                                                               \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; /* default device */
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);

    return EXIT_SUCCESS;
}
