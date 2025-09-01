```cuda
/*
Aim of the program:
Query and print the maximum pitch in bytes for a memory allocation.

Thinking:
To obtain the maximum allowable pitch for a 2D memory allocation on a CUDA device, the CUDA Runtime API provides the device attribute
cudaDevAttrMaxPitch. The function cudaDeviceGetAttribute can be used to query this value for each device. 
We first determine how many CUDA-capable devices are available using cudaGetDeviceCount. 
For each device, we call cudaDeviceGetAttribute with the attribute cudaDevAttrMaxPitch and print the result.
Error checking is performed after each CUDA API call to ensure robust operation. 
The output is printed in bytes and uses unsigned long long formatting to accommodate 64-bit values.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        size_t maxPitch = 0;
        err = cudaDeviceGetAttribute((int*)&maxPitch, cudaDevAttrMaxPitch, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Error: cudaDeviceGetAttribute for device %d failed: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: Max pitch for 2D allocation = %llu bytes\n", dev, (unsigned long long)maxPitch);
    }

    return 0;
}
```