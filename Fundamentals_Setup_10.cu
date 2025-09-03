/*
 * The aim of this program is to check and print the L2 cache size of the primary GPU in kilobytes (KB).
 *
 * Thinking:
 * 1. We will use CUDA Runtime API to query device properties.
 * 2. The relevant field in cudaDeviceProp is `l2CacheSize`, which gives the size in bytes.
 * 3. We'll query the first device (device 0) which is usually the primary GPU.
 * 4. Convert bytes to kilobytes by dividing by 1024.
 * 5. If `l2CacheSize` is zero (indicating not reported), we output an appropriate message.
 * 6. Include basic error checking after CUDA calls.
 * 7. The program will compile with `nvcc` and run on a system with CUDA installed.
 */

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the primary device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (prop.l2CacheSize == 0) {
        printf("L2 cache size is not reported for device %d.\n", device);
    } else {
        size_t l2CacheSizeKB = prop.l2CacheSize / 1024;
        printf("Device %d L2 cache size: %zu KB\n", device, l2CacheSizeKB);
    }

    return EXIT_SUCCESS;
}
