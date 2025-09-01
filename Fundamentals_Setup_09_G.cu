```c
/*
Aim of the program: Check if the device supports global L1 cache (`globalL1CacheSupported`).

Thinking process:
- The CUDA Runtime API provides a struct `cudaDeviceProp` which contains a field `globalL1CacheSupported` that indicates whether the device has a global L1 cache.
- The simplest way to check this is to query the device properties using `cudaGetDeviceProperties` for each device available on the system.
- For each device, we will print the device name and whether global L1 cache is supported.
- We will also handle errors from the CUDA API calls by checking the returned `cudaError_t` and printing an error message if needed.
- The program will be written in plain C (CUDA C) and will compile with `nvcc` to produce a .cu file. 
- Since the requirement is to output a .cu file only, the code below is self-contained and can be compiled and executed directly.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        if (prop.globalL1CacheSupported) {
            printf("  Global L1 cache is supported.\n");
        } else {
            printf("  Global L1 cache is NOT supported.\n");
        }
    }

    return 0;
}
```