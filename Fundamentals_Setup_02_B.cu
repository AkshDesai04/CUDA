/*
Aim: Iterate through all devices and, for each one, print its multiprocessor count.

Thinking:
- To get the number of CUDA devices available, we use cudaGetDeviceCount().
- For each device index, we retrieve its properties with cudaGetDeviceProperties().
- The property multiProcessorCount gives the number of multiprocessors.
- We'll print each device's index and its multiprocessor count.
- Basic error handling is added to report if the CUDA API calls fail.
- The program is self‑contained and can be compiled with nvcc.
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
        printf("No CUDA devices found.\n");
        return 0;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %d multiprocessors\n", dev, prop.multiProcessorCount);
    }

    return 0;
}
