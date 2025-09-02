/*
 * Aim: Query and print the concurrentKernels property.
 *
 * Thinking:
 * The CUDA runtime API provides cudaGetDeviceProperties to retrieve a
 * cudaDeviceProp structure for a given device index. The structure contains a
 * member named concurrentKernels which indicates whether concurrent kernel
 * execution is supported on the device. We want a simple host program that:
 * 1. Calls cudaGetDeviceCount to determine the number of devices.
 * 2. Iterates over each device index.
 * 3. Calls cudaGetDeviceProperties for that index.
 * 4. Prints the value of concurrentKernels in a human‑readable form.
 * The program should handle errors from the CUDA API calls gracefully by
 * printing an error message and exiting. It will use the standard
 * <cuda_runtime.h> header and compile with nvcc. No device kernels are needed.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d (%s): concurrentKernels = %s\n",
               dev,
               prop.name,
               prop.concurrentKernels ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
