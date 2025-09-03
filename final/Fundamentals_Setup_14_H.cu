```cpp
/*
Check if the device supports host-mapped pageable memory access (`pageableMemoryAccess`).

The goal of this program is to query the CUDA-capable devices present in the system
and determine whether each device has support for host-mapped pageable memory access.
This capability is indicated by the `pageableMemoryAccess` field in the
`cudaDeviceProp` structure returned by `cudaGetDeviceProperties`. If the field
is set to 1, the device can map pageable (non-pinned) host memory directly
into its address space, which can be useful for certain memory transfer
optimizations. If it is 0, the device does not support this feature.

To achieve this, the program performs the following steps:

1. Query the total number of CUDA devices using `cudaGetDeviceCount`.
2. Iterate over each device index.
3. For each device, call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
4. Inspect the `pageableMemoryAccess` field of the struct.
5. Print a human‑readable message indicating whether the feature is supported
   for that device.
6. Handle any CUDA API errors gracefully by printing an error message and
   exiting with a non‑zero status.

The program uses only the CUDA runtime API, so it can be compiled with `nvcc`
(e.g., `nvcc -o check_pageable_mem_access check_pageable_mem_access.cu`). No
external libraries or headers beyond `<stdio.h>` and `<cuda_runtime.h>` are
required. The output is straightforward and suitable for quick debugging
or system information gathering.

*/

// Include necessary headers
#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Found %d CUDA-capable device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error: cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        if (prop.pageableMemoryAccess)
        {
            printf("  Supports host-mapped pageable memory access.\n");
        }
        else
        {
            printf("  Does NOT support host-mapped pageable memory access.\n");
        }

        // Optional: print additional relevant properties
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total global memory: %llu MB\n",
               (unsigned long long)prop.totalGlobalMem / (1024 * 1024));
        printf("\n");
    }

    return 0;
}
```