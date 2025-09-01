/*
For each device, query and print whether it supports running kernels concurrently.

The program aims to retrieve the CUDA device count using cudaGetDeviceCount, iterate over each device index, and obtain the device properties via cudaGetDeviceProperties. The property cudaDeviceProp.concurrentKernels indicates whether the device supports concurrent kernel execution. We print a human-readable message for each device, reporting support or lack thereof.

The implementation uses the CUDA Runtime API in C. Error checking is performed after each CUDA call, printing an error message and terminating if a call fails. The program outputs to standard output, listing each device index, its name, and its concurrency capability.

The code is straightforward: include <cuda_runtime.h> for CUDA API, <stdio.h> for I/O. The main function obtains the device count, loops over devices, retrieves properties, checks the concurrentKernels field, and prints the result. If there are no CUDA devices, it prints a corresponding message. The program exits with status 0 on success or 1 on failure.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            return 1;
        }

        const char *concurrency = prop.concurrentKernels ? "supports" : "does NOT support";
        printf("Device %d: %s (%s) -- %s concurrent kernel execution.\n",
               dev, prop.name, prop.driverVersion ? "driver present" : "no driver",
               concurrency);
    }

    return 0;
}
