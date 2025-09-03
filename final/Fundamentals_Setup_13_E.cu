/*
Aim of the program: Query and print the device's multiprocessor count.

Thinking:
- The task is straightforward: use CUDA runtime API to get device properties and print the number of multiprocessors (SMs).
- We'll include <cuda_runtime.h> for CUDA API and <stdio.h> for printing.
- First, query the number of CUDA-capable devices using cudaGetDeviceCount.
- For each device, call cudaGetDeviceProperties to obtain struct cudaDeviceProp.
- The field 'multiProcessorCount' holds the number of SMs.
- Print the device ID and SM count.
- Handle errors: if any CUDA call fails, print an error message and exit.
- Use a simple main function; no GPU kernels needed.
- Compile with nvcc and run: nvcc -o device_sm_count device_sm_count.cu && ./device_sm_count
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
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
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d: \"%s\" has %d multiprocessor(s).\n",
               dev, prop.name, prop.multiProcessorCount);
    }

    return 0;
}
