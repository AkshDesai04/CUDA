/*
Aim: Modify the program to print the L2 cache size in megabytes (MB) with one decimal place.

Thinking:
1. Use CUDA runtime API to query device properties.
2. Retrieve the L2 cache size from cudaDeviceProp.l2CacheSize, which is given in bytes.
3. Convert the byte value to megabytes by dividing by 1024.0 twice.
4. Print the result with a single decimal place using printf("%.1f MB\n", ...).
5. Include basic error handling for CUDA API calls.
6. Keep the code minimal and self-contained, suitable for compiling as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

int main() {
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current CUDA device: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    size_t l2CacheBytes = prop.l2CacheSize;
    double l2CacheMB = (double)l2CacheBytes / (1024.0 * 1024.0);

    printf("L2 cache size: %.1f MB\n", l2CacheMB);

    return EXIT_SUCCESS;
}
