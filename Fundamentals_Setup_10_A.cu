```cuda
/*
 * Aim: Modify the program to print the L2 cache size in megabytes (MB) with one decimal place.
 *
 * Thinking:
 * 1. The CUDA runtime provides device properties through cudaGetDeviceProperties().
 *    Among those properties is `l2CacheSize`, which represents the size of the L2 cache in bytes.
 * 2. To express the size in megabytes with one decimal, we convert bytes to MB using the
 *    conversion factor 1 MB = 1024 * 1024 bytes.
 * 3. Use `float` or `double` for the conversion to preserve fractional values.
 * 4. Output the result with a single decimal using `printf("%.1f MB\n", sizeInMB);`.
 * 5. The program will query the first CUDA device (device 0). If no CUDA devices are
 *    available or an error occurs, we will print an error message and exit with a
 *    non‑zero status code.
 * 6. The code is self‑contained, includes necessary headers, and can be compiled with
 *    `nvcc` to produce an executable that prints the L2 cache size.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use device 0
    int device = 0;
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Convert L2 cache size from bytes to megabytes with one decimal
    double l2CacheSizeMB = prop.l2CacheSize / (1024.0 * 1024.0);
    printf("Device %d L2 Cache Size: %.1f MB\n", device, l2CacheSizeMB);

    return EXIT_SUCCESS;
}
```