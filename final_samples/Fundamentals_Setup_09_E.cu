/*
Iterate through all devices and print the `concurrentKernels` status for each one.

The goal of this program is to query every CUDA-capable device present on the system
and report whether it supports concurrent kernel execution. The `concurrentKernels`
field in the `cudaDeviceProp` structure is an integer flag that indicates this
capability: 1 if concurrent kernels are supported, 0 otherwise.

Approach:
1. Use `cudaGetDeviceCount` to determine how many devices are available.
2. Loop over each device index from 0 to count-1.
3. For each device, retrieve its properties using `cudaGetDeviceProperties`.
4. Extract the `concurrentKernels` field and print it along with the device
   name for clarity.
5. Perform basic error checking after each CUDA API call and report
   any failures in a user-friendly manner.

The program uses only the C CUDA Runtime API, so it can be compiled with `nvcc`
and runs on any system with a compatible CUDA toolkit installed.
It prints the result to standard output.
*/

// Include the CUDA runtime API and standard I/O.
#include <cuda_runtime.h>
#include <stdio.h>

// Helper macro to check CUDA errors.
#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                      \
        }                                                             \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  concurrentKernels = %d\n", prop.concurrentKernels);
    }

    return EXIT_SUCCESS;
}
