```cuda
/*
Check if the device supports global L1 cache (`globalL1CacheSupported`).

I want to produce a CUDA C program that enumerates the CUDA-capable devices on the system and prints whether each device supports a global L1 cache. The CUDA Runtime API provides a struct cudaDeviceProp that contains a boolean field globalL1CacheSupported. I'll use cudaGetDeviceCount to get the number of devices, then iterate, call cudaGetDeviceProperties for each device, and check the field. I'll also perform basic error checking on CUDA API calls. The program will print the device number, name, and the support status. That is the entire purpose of the program.

The code will be written in CUDA C, not C++ only but C style, using stdio.h for printing. I will not use any advanced features. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Global L1 Cache Supported: %s\n",
               prop.globalL1CacheSupported ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
```