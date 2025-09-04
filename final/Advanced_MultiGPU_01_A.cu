```cpp
/*
Aim of the program: Write a program that allocates 100MB of memory on every available GPU.

Thought process:
1. Query the number of CUDA-capable devices with cudaGetDeviceCount.
2. For each device:
   - Set the current device with cudaSetDevice.
   - Allocate 100 * 1024 * 1024 bytes (100 MB) using cudaMalloc.
   - Check for allocation success; if it fails, print the error and continue.
3. Store each allocated pointer in an array so that the allocations remain alive until program exit.
4. After allocating on all devices, print a summary and then exit.
   - Since the program terminates, the allocated device memory will be freed automatically by the OS/runtime.
5. Use a simple error-checking macro for clarity.
6. Keep the code in plain C (using CUDA Runtime API) and compile with nvcc as a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple macro for checking CUDA errors */
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return -1;                                             \
        }                                                          \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    /* Allocate an array to store pointers for each device */
    void **devicePointers = (void**)malloc(deviceCount * sizeof(void*));
    if (devicePointers == NULL) {
        fprintf(stderr, "Failed to allocate host memory for device pointers.\n");
        return -1;
    }

    const size_t allocSize = 100 * 1024 * 1024; /* 100 MB */

    for (int dev = 0; dev < deviceCount; ++dev) {
        printf("\nDevice %d:\n", dev);
        CHECK_CUDA(cudaSetDevice(dev));

        /* Optional: query device properties */
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
        printf("  Name: %s\n", prop.name);
        printf("  Total global memory: %lu bytes\n", prop.totalGlobalMem);

        /* Attempt to allocate 100 MB */
        cudaError_t allocErr = cudaMalloc(&devicePointers[dev], allocSize);
        if (allocErr == cudaSuccess) {
            printf("  Successfully allocated 100 MB on device %d.\n", dev);
        } else {
            printf("  Failed to allocate 100 MB on device %d: %s\n",
                   dev, cudaGetErrorString(allocErr));
            devicePointers[dev] = NULL;
        }
    }

    /* Keep allocations alive for a short period (optional) */
    // sleep(5); // Uncomment if you want the program to pause before exiting

    /* Clean up: free device memory */
    for (int dev = 0; dev < deviceCount; ++dev) {
        if (devicePointers[dev] != NULL) {
            cudaSetDevice(dev);
            cudaFree(devicePointers[dev]);
        }
    }

    free(devicePointers);

    printf("\nAll allocated memory has been freed. Exiting.\n");
    return 0;
}
```