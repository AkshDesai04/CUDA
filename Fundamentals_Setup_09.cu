```cpp
/*
Aim: Write a program to check and print whether the primary GPU supports running multiple kernels concurrently.
The property you are looking for is `concurrentKernels` in `cudaDeviceProp`.

Thinking:
- The CUDA Runtime API provides a structure `cudaDeviceProp` that contains many properties of a device, one of which is `concurrentKernels`.
- The property `concurrentKernels` is an integer flag that indicates whether the device can run multiple kernels concurrently.
- To query this property, we need to:
  1. Retrieve the device count with `cudaGetDeviceCount`.
  2. Select the primary device (device 0) for the check.
  3. Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure for that device.
  4. Read the `concurrentKernels` field and print a human‑readable message.
- Since the request is for a simple check, no kernel launches are required.
- We'll add minimal error handling: if any CUDA call fails, we print an error and exit.
- The program will be a single .cu file with standard includes and a `main` function.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // Use the first device (primary GPU)
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    if (prop.concurrentKernels) {
        printf("This GPU supports running multiple kernels concurrently.\n");
    } else {
        printf("This GPU does NOT support running multiple kernels concurrently.\n");
    }

    return EXIT_SUCCESS;
}
```