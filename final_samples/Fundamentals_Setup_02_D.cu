/*
Iterate through all devices and print the maximum number of threads per block for each.
The goal of this program is to query the CUDA runtime for the number of available GPU
devices on the system. For each device we retrieve its properties using cudaGetDeviceProperties
and extract the maxThreadsPerBlock field. We print the device index, name, and the maximum
threads per block value. Basic error handling is included to catch failures from the CUDA
runtime API calls. The program is written in C using the CUDA runtime API and can be
compiled with nvcc to produce an executable that reports the requested device
information.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Number of CUDA devices: %d\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    dev, cudaGetErrorString(err));
            continue;
        }
        printf("Device %d: %s, maxThreadsPerBlock = %d\n",
               dev, prop.name, prop.maxThreadsPerBlock);
    }

    return EXIT_SUCCESS;
}
