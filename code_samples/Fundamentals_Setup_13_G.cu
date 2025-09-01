/*
Query and print whether a kernel execution timeout is enabled on the device.

Thoughts:
1. The CUDA runtime API provides the attribute `cudaDevAttrKernelExecTimeoutEnabled` to query whether the device enforces a maximum execution time for kernels. 
2. To retrieve this attribute we use `cudaDeviceGetAttribute`. It requires the attribute enum, a pointer to an int, and the device number.
3. The program will first get the number of CUDA devices using `cudaGetDeviceCount`. If no devices are present, it will print a message and exit.
4. For each device, the program will query the attribute and print a human‑readable message indicating whether the timeout is enabled.
5. Simple error checking is implemented via a macro that prints the error string and exits on failure.
6. The code uses only standard headers and the CUDA runtime API, making it straightforward to compile with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        int timeoutEnabled = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&timeoutEnabled,
                                          cudaDevAttrKernelExecTimeoutEnabled,
                                          dev));

        /* Retrieve device name for better output */
        char name[256];
        CUDA_CHECK(cudaDeviceGetName(name, sizeof(name), dev));

        printf("Device %d (%s): Kernel execution timeout %s\n",
               dev,
               name,
               timeoutEnabled ? "ENABLED" : "DISABLED");
    }

    return EXIT_SUCCESS;
}
