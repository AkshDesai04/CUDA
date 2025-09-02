/*
Write a program that allocates 100MB of memory on every available GPU.

Thinking:
- The task is to write a CUDA C program that queries all GPUs available on the system using cudaGetDeviceCount().
- For each device, we will set the device context with cudaSetDevice().
- Allocate 100 megabytes of device memory using cudaMalloc().
- Store the device pointer for each device so we can keep the allocation alive until we decide to free it.
- Handle potential allocation failures by checking the return status of cudaMalloc() and printing an error message.
- After allocation, optionally keep the program running for a short period (or wait for user input) so that the allocations remain in effect.
- Finally, free the allocations with cudaFree() and reset the devices with cudaDeviceReset().
- We should include necessary headers, use proper error checking macros, and print informative messages.
- The program will compile as a single .cu file and can be built with nvcc.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> // for sleep

// Macro to check CUDA errors
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    // Allocate 100 MB per device
    const size_t allocSize = 100 * 1024 * 1024; // 100 MB in bytes

    // Arrays to hold device pointers and device IDs
    void **devicePtr = (void**)malloc(deviceCount * sizeof(void*));
    if (!devicePtr) {
        fprintf(stderr, "Failed to allocate host memory for device pointers.\n");
        return EXIT_FAILURE;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        CHECK_CUDA(cudaSetDevice(dev));
        printf("Allocating %zu bytes on device %d...\n", allocSize, dev);
        CHECK_CUDA(cudaMalloc(&devicePtr[dev], allocSize));
        printf("Allocation on device %d successful.\n", dev);
    }

    printf("\nAll allocations completed. The memory will remain allocated until the program exits.\n");
    printf("Press Enter to free memory and exit...\n");
    getchar();

    // Free allocations
    for (int dev = 0; dev < deviceCount; ++dev) {
        CHECK_CUDA(cudaSetDevice(dev));
        if (devicePtr[dev]) {
            CHECK_CUDA(cudaFree(devicePtr[dev]));
            printf("Freed memory on device %d.\n", dev);
        }
    }

    // Reset devices
    for (int dev = 0; dev < deviceCount; ++dev) {
        CHECK_CUDA(cudaSetDevice(dev));
        CHECK_CUDA(cudaDeviceReset());
    }

    free(devicePtr);
    printf("All resources cleaned up. Exiting.\n");
    return EXIT_SUCCESS;
}
