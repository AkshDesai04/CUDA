/*
For each device, print its maximum grid dimensions ([X, Y, Z]).

Thought Process:
- The goal is straightforward: iterate over all CUDA-capable devices available on the system
  and report each device's maximum grid dimensions in the X, Y, and Z directions.
- To achieve this, we use the CUDA Runtime API:
  * cudaGetDeviceCount() returns the number of CUDA devices.
  * cudaGetDeviceProperties() gives us a cudaDeviceProp structure that contains
    the maximum grid size in each dimension (maxGridSize[3]).
- We will print the results to the console using printf.
- Error handling is minimal; we check the return status of the CUDA API calls
  and abort with an error message if something goes wrong.
- This program is written in C with CUDA extensions, suitable for compilation with
  nvcc into a .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0)
    {
        printf("No CUDA devices found.\n");
        return 0;
    }

    printf("Found %d CUDA device(s):\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n", dev, cudaGetErrorString(err));
            continue;
        }

        printf("Device %d (%s): Max Grid Dimensions = [%d, %d, %d]\n",
               dev,
               prop.name,
               prop.maxGridSize[0],
               prop.maxGridSize[1],
               prop.maxGridSize[2]);
    }

    return 0;
}
