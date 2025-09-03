```c
/*
Aim of the program: Print the maximum y-dimension of a grid of blocks.

Thinking:
To determine the maximum y-dimension for a grid of CUDA blocks, we need to query the device properties
using cudaGetDeviceProperties. The device property struct, cudaDeviceProp, contains an array maxGridSize
where index 0 corresponds to the x-dimension limit, index 1 to the y-dimension limit, and index 2 to the z-dimension limit.
We will:
1. Set the CUDA device (default is 0, but we could allow for multiple devices).
2. Retrieve the device properties.
3. Extract the value at maxGridSize[1] (y-dimension limit).
4. Print that value to standard output.

This program is simple, uses only host code (no kernels), and demonstrates how to access hardware limits
through the CUDA Runtime API. It will compile with nvcc as a .cu file and run on a machine with a CUDA-capable GPU.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err;

    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum y-dimension of a grid of blocks on device %d: %d\n",
           device, prop.maxGridSize[1]);

    return 0;
}
```