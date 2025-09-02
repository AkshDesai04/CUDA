/*
Aim of the program: Iterate through all devices and, for each one, print its multiprocessor count.

Thinking:
The program should query the number of CUDA-capable devices available on the system using cudaGetDeviceCount.
Then, for each device index from 0 to count-1, we should retrieve the device properties with cudaGetDeviceProperties.
The relevant field for the number of multiprocessors is multiProcessorCount.
We will print the device index, the name (optional for clarity), and the multiprocessor count.
Error handling will be performed by checking the return value of CUDA runtime API calls and printing an error message if necessary.
The program will be a simple C/C++ CUDA (.cu) file that can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count (error %d: %s)\n", err, cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d (error %d: %s)\n", dev, err, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: %s, multiprocessor count = %d\n", dev, prop.name, prop.multiProcessorCount);
    }

    return 0;
}
