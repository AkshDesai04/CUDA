/*
Aim of the program:
Iterate through all devices and print the warp size for each one.

Thinking:
1. Need to query the number of CUDA-capable devices using cudaGetDeviceCount.
2. For each device index from 0 to count-1:
   a. Use cudaGetDeviceProperties to obtain a cudaDeviceProp struct.
   b. Extract the warpSize field.
   c. Print the device index, name, and warp size.
3. Handle errors from CUDA API calls; if any fail, print an error message and exit.
4. Use standard C/C++ I/O (printf) for simplicity.
5. The program will be compiled with nvcc and produces a single .cu file.
6. No external dependencies beyond the CUDA runtime library.
7. The code will be self-contained, with a single main function and necessary includes.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
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

    printf("Found %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s\n", dev, prop.name);
        printf("  Warp size: %d\n", prop.warpSize);
    }

    return 0;
}
