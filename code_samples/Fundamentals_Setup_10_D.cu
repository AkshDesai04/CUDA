```cuda
/*
Query and print the `persistingL2CacheMaxSize` property.

Thinking:
- The goal is to write a CUDA C program that queries the `persistingL2CacheMaxSize`
  field from the `cudaDeviceProp` structure and prints it.
- The program will:
  1. Determine the number of CUDA-capable devices.
  2. For each device, retrieve its properties with `cudaGetDeviceProperties`.
  3. Access the `persistingL2CacheMaxSize` member (available in newer CUDA
     versions) and print its value in bytes.
- Error handling will be performed for each CUDA runtime call to ensure
  graceful failure if something goes wrong (e.g., no devices, unsupported
  property).
- The output will list the device index, name, and the size of the
  persisting L2 cache.
- The program is a simple host-only program, no device code is required.
- It is written in a `.cu` file, which can be compiled with `nvcc`.

*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);
    printf("Querying persistingL2CacheMaxSize for each device:\n\n");

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        // Some older CUDA versions may not expose persistingL2CacheMaxSize.
        // We check its presence by ensuring the struct has the field.
        // In CUDA 12 and later, it's available.

        printf("Device %d: %s\n", dev, prop.name);
        printf("  persistingL2CacheMaxSize: %zu bytes\n\n",
               prop.persistingL2CacheMaxSize);
    }

    return EXIT_SUCCESS;
}
```