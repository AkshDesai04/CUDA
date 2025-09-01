```cuda
/*
Aim of the program: Check if the device supports stream priorities (streamPrioritiesSupported).

Thought process:
1. The CUDA Runtime API provides a property `streamPrioritiesSupported` in `cudaDeviceProp` that indicates whether a device supports stream priorities.
2. To query this property, we need to:
   - Get the number of CUDA devices with `cudaGetDeviceCount`.
   - For each device (or just device 0 for simplicity) retrieve its properties using `cudaGetDeviceProperties`.
3. After obtaining `cudaDeviceProp`, we can inspect `streamPrioritiesSupported` (a boolean flag).
4. We will print a message indicating whether stream priorities are supported.
5. Include error handling for all CUDA API calls: check return codes and print appropriate error messages.
6. The program will be written in C/C++ with CUDA extensions and compile with `nvcc`.

This program demonstrates a simple, robust query of a device capability and provides clear output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return 1;
    }

    // For simplicity, query device 0. Adjust if you want to iterate over all devices.
    int dev = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Device %d: \"%s\"\n", dev, prop.name);
    if (prop.streamPrioritiesSupported) {
        printf("This device supports stream priorities.\n");
    } else {
        printf("This device does NOT support stream priorities.\n");
    }

    return 0;
}
```