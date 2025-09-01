/*
Aim: Iterate through all available GPUs and print the L2 cache size for each.

Thinking:
The program must query the number of CUDA devices present on the system, then iterate over each device
to retrieve its properties. The CUDA runtime API provides `cudaGetDeviceCount` to obtain the count and
`cudaGetDeviceProperties` to obtain a `cudaDeviceProp` structure for each device index. This structure
includes the field `l2CacheSize`, which reports the size of the L2 cache in bytes. The program will print
a line for each GPU, showing the device index, its name, and the L2 cache size. Basic error handling
checks are added to handle cases where the device count query fails or properties cannot be fetched.
The program is written in plain C with CUDA runtime headers and can be compiled with `nvcc` as a
standalone executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount returned %d (%s)\n",
                err, cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties for device %d failed (%s)\n",
                    i, cudaGetErrorString(err));
            continue;
        }

        printf("GPU %d: %s\n", i, prop.name);
        printf("  L2 cache size: %zu bytes\n", prop.l2CacheSize);
    }

    return 0;
}
