/*
Query and print the maximum number of threads per multiprocessor.
This program demonstrates how to query device properties in CUDA and specifically retrieve the maximum number of threads that can be resident on a single streaming multiprocessor (SM). The steps involved are:
1. Determine the number of CUDA-capable devices available.
2. For each device, obtain a cudaDeviceProp structure via cudaGetDeviceProperties().
3. Extract the maxThreadsPerMultiProcessor field from the structure.
4. Print the value along with the device name for clarity.
The program includes basic error checking after each CUDA API call to ensure that any failure is reported and the program exits gracefully. This example can be compiled with `nvcc` and executed on any system with a CUDA-capable GPU. It serves as a minimal example of querying hardware capabilities in a CUDA application.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                    \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Max threads per multiprocessor: %d\n\n",
               prop.maxThreadsPerMultiProcessor);
    }

    return EXIT_SUCCESS;
}
