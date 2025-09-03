/*
Iterate through all devices and print the maximum number of threads per block for each.

Thinking:
The program needs to query the number of CUDA-capable devices available on the system using cudaGetDeviceCount. 
Once we have that count, we iterate over each device index from 0 to count-1. For each device, we retrieve its properties
by calling cudaGetDeviceProperties, which fills a cudaDeviceProp struct with various device details. 
The maximum number of threads that can be launched per block is stored in the maxThreadsPerBlock field of this struct. 
We'll print the device index, the device name, and this maximum value. 
Basic error checking is performed after each CUDA runtime API call to ensure proper operation. 
The program uses standard C headers and CUDA runtime header <cuda_runtime.h> and is compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
    }

    return 0;
}
