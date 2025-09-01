/*
Check if the device supports local L1 cache (`localL1CacheSupported`).

Aim of the program (word to word as passed in the prompt):
Check if the device supports local L1 cache (`localL1CacheSupported`).

Thinking:
- Use the CUDA Runtime API to query device properties.
- First get the number of CUDA-capable devices with cudaGetDeviceCount.
- Iterate over each device index.
- For each device, call cudaGetDeviceProperties to fill a cudaDeviceProp struct.
- The struct contains a boolean field `localL1CacheSupported` (available in CUDA 9.0+).
- Print out whether each device supports local L1 cache.
- Include basic error handling for the CUDA API calls.
- Output results to standard output.
- Ensure the code compiles with nvcc as a single .cu file.
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
        if (prop.localL1CacheSupported) {
            printf("  Supports local L1 cache: Yes\n");
        } else {
            printf("  Supports local L1 cache: No\n");
        }
    }

    return 0;
}
