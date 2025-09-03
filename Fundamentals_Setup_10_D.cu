```cuda
/* 
Aim of the program:
Query and print the `persistingL2CacheMaxSize` property.

Thinking:
- The property `persistingL2CacheMaxSize` is exposed via the CUDA Runtime API
  as the device attribute `cudaDevAttrPersistingL2CacheMaxSize`.  
- We will use `cudaGetDeviceCount()` to determine the number of CUDA-capable
  devices present on the system.  
- For each device, we will call `cudaDeviceGetAttribute()` with the attribute
  identifier to retrieve the size in bytes.  
- The returned value is an integer; if the attribute is not supported on a
  particular device the API will return `cudaErrorInvalidValue`.  
- We will print the device index and the attribute value (or an error
  message) in a human-readable format.
- Error handling will be performed with a helper macro `CUDA_CHECK` that
  prints an error message and exits if any CUDA API call fails.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Helper macro for error checking
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        int persistingL2CacheMaxSize = 0;
        cudaError_t err = cudaDeviceGetAttribute(&persistingL2CacheMaxSize,
                                                 cudaDevAttrPersistingL2CacheMaxSize,
                                                 dev);
        if (err == cudaSuccess) {
            printf("Device %d: persistingL2CacheMaxSize = %d bytes\n",
                   dev, persistingL2CacheMaxSize);
        } else if (err == cudaErrorInvalidValue) {
            printf("Device %d: persistingL2CacheMaxSize attribute not supported.\n",
                   dev);
        } else {
            fprintf(stderr, "Failed to query persistingL2CacheMaxSize for device %d: %s\n",
                    dev, cudaGetErrorString(err));
        }
    }

    return EXIT_SUCCESS;
}
```