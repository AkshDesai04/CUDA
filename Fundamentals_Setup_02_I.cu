```cpp
/*
Aim of the program: For each device, query and print whether it supports running kernels concurrently.

Thinking:
- The goal is to examine every CUDA-capable device present on the system and determine if it can run kernels concurrently. 
- CUDA provides the device property 'concurrentKernels' in the cudaDeviceProp structure, which is set to 1 if the device supports concurrent kernel execution.
- We need to query the number of devices using cudaGetDeviceCount.
- For each device, we retrieve its properties with cudaGetDeviceProperties.
- We then check the 'concurrentKernels' field and print an appropriate message.
- Basic error checking is added for CUDA API calls to ensure robustness.
- The output will display the device index, name, and whether concurrent kernel execution is supported.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        if (prop.concurrentKernels) {
            printf("  Supports running kernels concurrently.\n");
        } else {
            printf("  Does NOT support running kernels concurrently.\n");
        }
    }

    return EXIT_SUCCESS;
}
```