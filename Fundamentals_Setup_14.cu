```cuda
/*
Write a program to check if the primary GPU can overlap data copying with kernel execution. This capability is crucial for performance and is indicated by the asyncEngineCount property (a value > 0).

--- Thinking ---
To determine if the primary GPU supports overlapping data transfers with kernel execution, we need to query the device properties for the property `asyncEngineCount`. If this count is greater than zero, the device has at least one asynchronous engine that can perform concurrent copy and execution. The program will:

1. Include the necessary CUDA runtime header.
2. Define a simple error-checking macro to wrap CUDA API calls.
3. In `main`, query the number of devices; exit if none.
4. Select the primary device (device 0).
5. Retrieve its properties with `cudaGetDeviceProperties`.
6. Print the value of `asyncEngineCount`.
7. Indicate whether overlapping is supported based on the value.

This is a straightforward diagnostic utility that can be compiled with `nvcc` and executed on a system with CUDA installed.

Note: No actual kernel or copy operations are performed; we only rely on the property value. If needed, one could extend the program to launch a kernel and perform async copies to empirically verify overlap, but that is beyond the scope of this simple check.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Simple macro for error checking */
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return EXIT_FAILURE;
    }

    /* Use the primary device (device 0) */
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));

    printf("Device %d: %s\n", device, prop.name);
    printf("  asyncEngineCount: %d\n", prop.asyncEngineCount);

    if (prop.asyncEngineCount > 0) {
        printf("  Overlap of data copy and kernel execution is supported.\n");
    } else {
        printf("  Overlap of data copy and kernel execution is NOT supported.\n");
    }

    return EXIT_SUCCESS;
}
```