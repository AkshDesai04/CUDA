/*
 * Aim: Write a program to check and print the L2 cache size of the primary GPU in kilobytes (KB).
 *
 * Thinking:
 * - The program uses the CUDA Runtime API to query the properties of the primary GPU.
 * - We first determine how many CUDA devices are available with cudaGetDeviceCount.
 * - The primary GPU is considered to be device 0; if no device is present, we report an error.
 * - The cudaDeviceProp structure returned by cudaGetDeviceProperties contains a field
 *   l2CacheSize which represents the size of the L2 cache in bytes. This field is
 *   supported on GPUs of compute capability 5.0 and newer. For older GPUs it may be
 *   zero or undefined, so we handle that case by reporting "not available" if the
 *   size is zero.
 * - We convert the size from bytes to kilobytes by dividing by 1024 and print the
 *   result in a human‑readable format.
 * - Basic error checking is performed after each CUDA call; if an error occurs we
 *   print the error string using cudaGetErrorString and exit with a non‑zero status.
 *
 * The program is a straightforward console application and can be compiled with
 * nvcc as follows:
 *   nvcc -o l2_cache_size l2_cache_size.cu
 *
 * Example output on a GPU with an 8 MB L2 cache:
 *   L2 cache size of device 0: 8192 KB
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // Primary device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    size_t l2CacheBytes = prop.l2CacheSize; // in bytes
    if (l2CacheBytes == 0) {
        printf("L2 cache size of device %d: Not available or zero bytes.\n", device);
    } else {
        size_t l2CacheKB = l2CacheBytes / 1024;
        printf("L2 cache size of device %d: %zu KB\n", device, l2CacheKB);
    }

    return EXIT_SUCCESS;
}
