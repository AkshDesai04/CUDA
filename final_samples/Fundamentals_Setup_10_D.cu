/*
Aim of the program: Query and print the persistingL2CacheMaxSize property.

Thinking process:
1. The `persistingL2CacheMaxSize` is a field of the `cudaDeviceProp` structure, introduced in recent CUDA versions to report the maximum size (in bytes) of the persisting L2 cache that a device can use. We need to retrieve this value for each available CUDA device.
2. The program will:
   - Include the necessary headers: `<cuda_runtime.h>` and `<stdio.h>`.
   - Call `cudaGetDeviceCount()` to find how many CUDA-capable devices are present.
   - For each device, call `cudaGetDeviceProperties()` to fill a `cudaDeviceProp` structure.
   - Access the `persistingL2CacheMaxSize` field from that structure.
   - Print the value in a human‑readable format (bytes and, optionally, megabytes).
3. Error checking will be performed after each CUDA API call. If any call fails, the program will print an error message and exit.
4. The code is written in CUDA C and will compile into a `.cu` file. It uses only runtime API functions, so no separate device kernels are needed.
5. The output will list each device’s index, name, and persisting L2 cache max size in bytes (and in MB for convenience).

The final program is self‑contained and ready to compile with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Persisting L2 Cache Max Size: %lu bytes (%.2f MB)\n\n",
               (unsigned long)prop.persistingL2CacheMaxSize,
               (double)prop.persistingL2CacheMaxSize / (1024 * 1024));
    }

    return EXIT_SUCCESS;
}
