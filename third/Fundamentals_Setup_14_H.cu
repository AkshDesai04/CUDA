/*
Check if the device supports host-mapped pageable memory access (`pageableMemoryAccess`).

The goal of this CUDA C program is to query each CUDA-capable device present on the system
and determine whether it supports pageable memory access. This capability is represented by
the device attribute `cudaDevAttrPageableMemoryAccess` available in the CUDA runtime API.
The program performs the following steps:

1. Use `cudaGetDeviceCount()` to obtain the number of CUDA devices.
2. Iterate over each device index.
3. For each device, call `cudaDeviceGetAttribute()` with the attribute `cudaDevAttrPageableMemoryAccess`.
4. Interpret the returned value: a non-zero value indicates that the device supports
   host-mapped pageable memory access; zero indicates it does not.
5. Print a readable message for each device, reporting the support status.

The program includes minimal error checking: it reports and aborts if any CUDA API call
fails. This is sufficient for a simple diagnostic tool and keeps the code concise.

This approach directly uses the CUDA Runtime API, which is the most straightforward way
to query device attributes from a host program. The code is written in standard C
compatible with CUDA’s compiler (`nvcc`) and compiles into a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        int pageableSupport = 0;
        err = cudaDeviceGetAttribute(&pageableSupport,
                                     cudaDevAttrPageableMemoryAccess,
                                     dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting attribute for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            return 1;
        }

        const char *supportText = pageableSupport ? "YES" : "NO";
        printf("Device %d supports pageable memory access: %s\n", dev, supportText);
    }

    return 0;
}
