/*
Aim of the program: A more direct property is `deviceOverlap`. Query and print this boolean property for the primary device.

Thinking:
To achieve this, we will write a simple CUDA host program in C that queries the `cudaDeviceProp` structure for the primary device (device 0). The `deviceOverlap` field of this structure indicates whether the device can simultaneously execute kernels and copy memory. Since this field is an integer (0 or 1), we will cast it to a boolean for clarity when printing. 

The program flow will be:
1. Include the necessary headers (`<stdio.h>` and `<cuda_runtime.h>`).
2. Define a helper macro `CHECK_CUDA` for error checking after CUDA API calls.
3. In `main`, call `cudaGetDeviceCount` to confirm at least one device is present.
4. Use `cudaGetDeviceProperties` to retrieve the properties for device 0.
5. Print the value of `deviceOverlap` as either `true` or `false`.
6. Return 0.

We will ensure the code compiles as a `.cu` file, so no CUDA kernel is needed. The program is self‑contained and prints the required boolean property. */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            return EXIT_FAILURE;                                          \
        }                                                                  \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    int dev = 0; // primary device
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    bool overlap = (prop.deviceOverlap != 0);
    printf("deviceOverlap for device %d: %s\n", dev, overlap ? "true" : "false");

    return EXIT_SUCCESS;
}
