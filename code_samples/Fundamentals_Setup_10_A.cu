/*
Modify the program to print the L2 cache size in megabytes (MB) with one decimal place.
The goal is to use CUDA Runtime API to query device properties, specifically the
l2CacheSize field which gives the size in bytes.  Converting this to megabytes
requires dividing by (1024 * 1024).  We then print the value with one decimal
precision using printf's "%.1f".  The program iterates over all CUDA devices,
fetches their properties, and outputs both the name and the L2 cache size in MB.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
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
            fprintf(stderr, "Error getting properties for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        double l2CacheMB = static_cast<double>(prop.l2CacheSize) / (1024.0 * 1024.0);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  L2 cache size: %.1f MB\n", l2CacheMB);
    }

    return 0;
}
