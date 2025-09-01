/*
Aim of the program:
Iterate through all available GPUs and print the L2 cache size for each.

Thinking:
To fulfill the task, the program needs to query the number of CUDA-capable GPUs present in the system using cudaGetDeviceCount. Then, for each device index, we retrieve its properties via cudaGetDeviceProperties, which returns a cudaDeviceProp structure. This structure contains the field l2CacheSize, representing the L2 cache size in bytes. We will print this value in a human-readable format. Error handling will be minimal: if cudaGetDeviceCount or cudaGetDeviceProperties fails, the program will output an error message and exit. The code is written in C/C++ style suitable for a .cu file, and uses standard headers and CUDA runtime API. No additional external libraries or helper files are needed. This keeps the implementation concise while meeting the requirement to output the L2 cache size for each GPU found.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties for device %d failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: \"%s\"\n", dev, prop.name);
        printf("  L2 cache size: %u bytes\n", prop.l2CacheSize);
    }

    return 0;
}
