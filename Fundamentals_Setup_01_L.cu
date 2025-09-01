```cpp
/*
Aim: Query and print the maximum number of threads that can be active on a single multiprocessor for the primary device.

Thinking:
- The CUDA runtime provides the cudaDeviceProp structure via cudaGetDeviceProperties.
- The field maxThreadsPerMultiProcessor (or maxThreadsPerMP in some older SDKs) gives the maximum number of active threads per SM.
- Since the problem asks for the "primary device", we will query device 0.
- A simple error‑checking helper macro is used to catch CUDA API errors.
- The program prints the value to stdout and returns 0 on success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            return EXIT_FAILURE;                                               \
        }                                                                      \
    } while (0)

int main(void) {
    int device = 0;  // primary device
    cudaDeviceProp prop;

    // Get device properties
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    // Print the maximum number of active threads per multiprocessor
    printf("Maximum number of threads that can be active on a single multiprocessor "
           "for device %d: %d\n", device, prop.maxThreadsPerMultiProcessor);

    return 0;
}
```