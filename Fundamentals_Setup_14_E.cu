/*
Aim: Query and print the `concurrentKernels` property.

The goal of this program is to demonstrate how to use the CUDA Runtime API to query device
properties and specifically retrieve the `concurrentKernels` flag, which indicates whether
the GPU can execute multiple kernels concurrently. The program performs the following steps:

1. Retrieve the number of CUDA-capable devices available on the system.
2. Iterate over each device, fetching its `cudaDeviceProp` structure.
3. Extract the `concurrentKernels` field from the structure.
4. Print the device index, name, and the value of `concurrentKernels` (0 for false, 1 for true).

The program is written in standard C++ (but uses only C-style code) and is intended to be
compiled with NVIDIA's NVCC compiler. Error checking is included to handle potential CUDA
runtime API failures gracefully. The output is printed to standard output.

This minimal example is useful for developers who need to confirm whether their target
hardware supports concurrent kernel execution, which can influence design decisions
around kernel launch strategies and resource scheduling.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Concurrent Kernels Supported: %s\n\n",
               prop.concurrentKernels ? "Yes" : "No");
    }

    return 0;
}
