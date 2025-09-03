```cuda
/*
Iterate through all devices and print the warp size for each one.

This program aims to query the number of CUDA-capable devices present on the system
and, for each device, retrieve its properties using cudaGetDeviceProperties.
The warpSize field in the cudaDeviceProp structure indicates the size of a
warp (the number of threads that execute in lockstep on an NVIDIA GPU).
The code will print the warp size for each device in a readable format.

Key steps:
1. Use cudaGetDeviceCount to determine how many devices exist.
2. Loop over each device index.
3. For each device, call cudaGetDeviceProperties to obtain a cudaDeviceProp
   structure containing hardware details.
4. Extract the warpSize from this structure and output it.
5. Handle errors from CUDA API calls gracefully by printing an error message
   and exiting if necessary.

The program compiles with `nvcc` and produces a single executable that
displays the warp size for each detected GPU.
*/
#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices found.\n");
        return 0;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d (%s): warp size = %d\n", dev, prop.name, prop.warpSize);
    }

    return 0;
}
```