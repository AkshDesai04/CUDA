```c
/*
Aim: For each device, print its L2 cache size in megabytes (MB).

Thinking process:
1. The goal is straightforward: query all CUDA-capable devices, retrieve the
   L2 cache size for each, and output it in megabytes.
2. CUDA provides `cudaGetDeviceCount` to know how many devices exist and
   `cudaGetDeviceProperties` to obtain a `cudaDeviceProp` structure with
   all device attributes.
3. The L2 cache size is stored in the `l2CacheSize` field of
   `cudaDeviceProp` (bytes).  We will convert this to MB by dividing by
   1024*1024.  If the field is zero (unknown), we simply print 0 MB.
4. The program will print:
   - Total device count
   - For each device: its index, name, and L2 cache size in MB.
5. Basic error handling will be added: if any CUDA call fails, the
   error string will be printed and the program will exit or skip the
   problematic device.
6. The code is written in plain C with CUDA runtime API, compiled with
   `nvcc`.  It requires no additional dependencies beyond the CUDA SDK.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }

        size_t l2SizeBytes = prop.l2CacheSize;   // size in bytes
        double l2SizeMB = (double)l2SizeBytes / (1024.0 * 1024.0);

        printf("Device %d: %s\n", dev, prop.name);
        printf("  L2 Cache Size: %.2f MB\n", l2SizeMB);
    }

    return EXIT_SUCCESS;
}
```