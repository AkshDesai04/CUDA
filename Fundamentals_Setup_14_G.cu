/*
Aim of the program:
Check if the device can map host memory using the `canMapHostMemory` property and print the result.

Thought process:
1. Include <cuda_runtime.h> for CUDA runtime API.
2. In main, query the number of CUDA devices using cudaGetDeviceCount.
3. If no devices, print a message and exit.
4. Loop over each device index:
   a. Retrieve cudaDeviceProp using cudaGetDeviceProperties.
   b. Examine the `canMapHostMemory` field.
   c. Print the result in a human-readable form ("yes" or "no").
5. Check for CUDA errors after each API call; if any error occurs, print the error string and exit with non-zero status.
6. Ensure that the program compiles as a .cu file and can be built with nvcc.

This program demonstrates how to query a specific device property and interpret the result.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaGetDeviceProperties for device %d failed: %s\n", dev, cudaGetErrorString(err));
            return 1;
        }

        printf("Device %d: %s (Name: %s)\n", dev,
               prop.canMapHostMemory ? "can map host memory" : "cannot map host memory",
               prop.name);
    }

    return 0;
}
