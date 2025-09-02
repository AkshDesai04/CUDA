/*
For each device, query and print whether it supports running kernels concurrently.
---

**Thinking:**

1. **Goal**  
   We need to inspect every CUDA-enabled device in the system and determine if it
   can launch multiple kernels concurrently. This capability is exposed through
   the CUDA Runtime API as the `CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS`
   attribute.

2. **Approach**  
   - Use `cudaGetDeviceCount()` to know how many GPUs are present.
   - For each device index `i`:
     - Retrieve its properties with `cudaGetDeviceProperties()`.  
       This gives us the device name and other useful info.
     - Query the `CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS` attribute via
       `cudaDeviceGetAttribute()`.  
       The value is 1 if concurrent kernels are supported, otherwise 0.
   - Print a readable line for each device: device number, name, and the
     support status.

3. **Error handling**  
   Each CUDA call returns a `cudaError_t`. We'll check the result and print
   a meaningful error message if something goes wrong, then exit.

4. **Implementation details**  
   - Include `<cuda_runtime.h>` and `<stdio.h>`.
   - Keep the code in pure C/C++ style; no additional libraries are required.
   - The program will compile with `nvcc` and produce an executable that
     prints the information to stdout.

5. **Edge cases**  
   - No devices found: print a message and exit gracefully.
   - Unsupported CUDA version or driver: error will be caught by the error
     checking logic.

This straightforward program satisfies the requirement: for every CUDA device
determine and print whether it supports concurrent kernel execution.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Detected %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

        int concurrentKernels = 0;
        CHECK_CUDA(cudaDeviceGetAttribute(&concurrentKernels,
                                          CUDA_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS,
                                          dev));

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Supports concurrent kernels: %s\n\n",
               concurrentKernels ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
