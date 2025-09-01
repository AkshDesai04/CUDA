```c
/*
Check if the device can map host memory using the `canMapHostMemory` property and print the result.
I started by deciding that the simplest way to determine if a CUDA device can map host memory is to query its device properties using the cudaGetDeviceProperties function. The canMapHostMemory field in the cudaDeviceProp structure indicates whether the device supports mapping host memory into the device address space. I planned to retrieve the properties for the default device (device 0), check the canMapHostMemory flag, and print a user-friendly message. 

To make the program robust, I included basic error checking for CUDA API calls: a macro CHECK_CUDA that reports errors and exits. I also added logic to handle the case when no CUDA devices are available. The output is intentionally minimal: it prints the device index and whether it supports mapping host memory. This straightforward approach keeps the code concise while fulfilling the requirement. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  canMapHostMemory: %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
```