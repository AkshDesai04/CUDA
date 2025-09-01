/*
Aim: Query and print the maximum z-dimension of a thread block.

Thinking:
The goal is to retrieve the maximum size of a thread block in the z-dimension for a CUDA-capable GPU.
The CUDA Runtime API provides the cudaDeviceProp struct, which contains the array maxThreadsDim[3].
index 0 → x-dimension, index 1 → y-dimension, index 2 → z-dimension.
We'll:
1. Use cudaGetDeviceProperties to obtain the properties for the default device (device 0).
2. Print the device name and the value of maxThreadsDim[2].
3. Handle any errors from the CUDA API calls with proper error messages.
This minimal program will compile with nvcc and run on any system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Device %d: %s\n", device, prop.name);
    printf("Maximum z-dimension of a thread block: %d\n", prop.maxThreadsDim[2]);

    return 0;
}
