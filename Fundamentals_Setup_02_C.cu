```c
/*
Iterate through all devices and print the warp size for each one.

This program demonstrates how to query CUDA device properties in a
simple C/C++ CUDA application. The goal is to retrieve the warp size
for each CUDA-capable device present on the system. The steps are
straightforward:

1. Include the necessary headers: <stdio.h> for I/O and <cuda_runtime.h>
   for the CUDA Runtime API.
2. Use cudaGetDeviceCount to determine how many devices are available.
3. Loop over each device index, call cudaGetDeviceProperties to obtain a
   cudaDeviceProp structure containing various attributes, including
   warpSize.
4. Print the device number, name, total global memory, and the warp size.
   Printing the name and memory gives context and verifies the device
   enumeration works.
5. Handle errors: If any CUDA API call fails, print the error string and
   exit with a non-zero status.

The program compiles with nvcc, e.g.:
    nvcc -o warp_sizes warp_sizes.cu
Running the executable will output lines like:
    Device 0: GeForce GTX 1080 Ti, 11178 MB, warp size = 32
    Device 1: CPU Device, 0 MB, warp size = 32
The exact output depends on the system's CUDA-enabled GPUs.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total global memory: %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
        printf("  Warp size: %d\n\n", prop.warpSize);
    }

    return 0;
}
```