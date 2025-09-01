/*
Aim: Query and print the `persistingL2CacheMaxSize` property.

Thinking:
The CUDA runtime provides the `cudaDeviceProp` structure via `cudaGetDeviceProperties`. This struct contains various hardware properties of a GPU device, including the size of the persisting L2 cache (field name `persistingL2CacheMaxSize`). 
To access this property we need to:
1. Query the number of CUDA-capable devices with `cudaGetDeviceCount`.
2. For each device, retrieve its properties using `cudaGetDeviceProperties`.
3. Print the device index, name, and the `persistingL2CacheMaxSize` field.
4. Handle any CUDA errors with a simple helper macro `CUDA_CHECK`.

The code below is a minimal, self‑contained CUDA program that performs these steps and prints the result to standard output. Compile with `nvcc -o query_l2_cache query_l2_cache.cu`. If your CUDA headers do not expose `persistingL2CacheMaxSize`, the code will fail to compile; this property is available starting from CUDA 11.2 or newer. If you run this on a device that does not support persisting L2 cache, the value will be reported as 0. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            return EXIT_FAILURE;                                   \
        }                                                           \
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
        printf("  persistingL2CacheMaxSize: %d bytes\n\n", prop.persistingL2CacheMaxSize);
    }

    return EXIT_SUCCESS;
}
