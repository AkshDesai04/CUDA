```cuda
/*
 * Query and check if the device supports Page-locked Memory Mapped On The GPU (`pageableMemoryAccess`).
 *
 * Thinking:
 *   - CUDA exposes the attribute CUDA_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS (since CUDA 9) to indicate
 *     whether a device can map pageable (non‑pinned) host memory into its address space, allowing zero‑copy
 *     access from the GPU.  If the attribute is 1, the device supports this feature; otherwise it is 0.
 *   - To determine this for the current device(s), we use cudaDeviceGetAttribute().
 *   - We will query the attribute for all devices present in the system.  For each device we report
 *     whether it supports pageable memory access.
 *   - Error handling is performed via a simple macro that checks the return value of CUDA runtime
 *     calls and aborts if an error occurs.
 *   - The program uses standard C functions (printf) and the CUDA Runtime API; it is a single .cu file
 *     that can be compiled with nvcc.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                    \
    do {                                                    \
        cudaError_t err = (call);                           \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,             \
                    cudaGetErrorString(err));               \
            return EXIT_FAILURE;                            \
        }                                                   \
    } while (0)

int main(void)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        printf("No CUDA-capable device detected.\n");
        return EXIT_SUCCESS;
    }

    printf("Detected %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        int pageableMemoryAccess = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&pageableMemoryAccess,
                                          cudaDevAttrPageableMemoryAccess,
                                          dev));

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("\nDevice %d: %s\n", dev, prop.name);
        printf("  Supports pageable memory access: %s\n",
               pageableMemoryAccess ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
```