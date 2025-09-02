```cpp
/*
Aim of the program: 
Iterate through all available CUDA devices and print their total global memory in gigabytes (GiB) for each.

Thinking:
1. To determine how many CUDA-capable devices are present, use cudaGetDeviceCount().
2. For each device index from 0 to count-1, retrieve the device properties using cudaGetDeviceProperties().
3. The total global memory size is available in the property struct as totalGlobalMem (in bytes).
4. Convert this size to GiB by dividing by 1024^3. 
5. Print the device index and its memory in a readable format.
6. Handle potential errors from CUDA API calls and provide useful output if no devices are found.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return EXIT_SUCCESS;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        double memGiB = (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Total Global Memory: %.3f GiB\n", memGiB);
    }

    return EXIT_SUCCESS;
}
```