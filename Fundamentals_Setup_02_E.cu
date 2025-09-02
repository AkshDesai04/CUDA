/*
Aim: Write a program that finds the device with the most global memory and prints its name and index.

Thinking:
- We need to query all CUDA-capable devices on the system.
- Use `cudaGetDeviceCount` to get the number of devices.
- Iterate over each device index, retrieve its properties with `cudaGetDeviceProperties`.
- Compare `totalGlobalMem` field of `cudaDeviceProp` to keep track of the device with the largest global memory.
- After the loop, print the name and index of the best device.
- Include basic error checking for CUDA API calls.
- Since this is a simple host-only program, no device kernels are required.
- The program will be a single .cu file that can be compiled with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    int bestDevice = -1;
    size_t maxGlobalMem = 0;
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        if (prop.totalGlobalMem > maxGlobalMem) {
            maxGlobalMem = prop.totalGlobalMem;
            bestDevice = dev;
        }
    }

    if (bestDevice >= 0) {
        cudaDeviceProp bestProp;
        err = cudaGetDeviceProperties(&bestProp, bestDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for best device %d: %s\n", bestDevice, cudaGetErrorString(err));
            return EXIT_FAILURE;
        }
        printf("Device with most global memory: %s (Index %d)\n", bestProp.name, bestDevice);
        printf("Total global memory: %.2f GB\n", (double)bestProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    } else {
        printf("Could not determine device with most global memory.\n");
    }

    return EXIT_SUCCESS;
}
