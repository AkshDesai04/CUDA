```c
/*
Aim of the program:
Iterate through all devices and print the warp size for each one.

Thinking:
1. CUDA provides an API to query the number of devices via cudaGetDeviceCount.
2. For each device index from 0 to count-1, we can retrieve a cudaDeviceProp struct using cudaGetDeviceProperties.
3. The warp size is stored in the warpSize field of the struct.
4. We should handle potential errors from the CUDA API calls; if any call fails, print an error message and exit.
5. The program will simply print the device index and its warp size to standard output.
6. No additional functionality is required. We'll include a small helper macro to check CUDA errors for brevity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = (call);                                \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void) {
    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));
        printf("Device %d: warpSize = %d\n", dev, prop.warpSize);
    }

    return EXIT_SUCCESS;
}
```