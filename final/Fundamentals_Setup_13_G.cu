```cpp
/*
Query and print whether a kernel execution timeout is enabled on the device.

My thoughts while writing this program:
- CUDA provides a device attribute called cudaDevAttrKernelExecTimeoutEnabled that indicates whether the device has a kernel execution timeout.
- The attribute can be queried using cudaDeviceGetAttribute(&value, attr, device).
- We'll query this attribute for the default device (device 0) and print a human‑readable message.
- To make the code robust, I'll include basic error checking for CUDA API calls.
- The program will be a minimal .cu file that compiles with nvcc and runs on any GPU that supports the attribute.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            return EXIT_FAILURE;                                            \
        }                                                                    \
    } while (0)

int main(void)
{
    // Query number of CUDA devices
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    // We'll use device 0
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    int timeoutEnabled = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&timeoutEnabled,
                                      cudaDevAttrKernelExecTimeoutEnabled,
                                      device));

    if (timeoutEnabled) {
        printf("Kernel execution timeout is enabled on device %d.\n", device);
    } else {
        printf("Kernel execution timeout is disabled on device %d.\n", device);
    }

    return EXIT_SUCCESS;
}
```