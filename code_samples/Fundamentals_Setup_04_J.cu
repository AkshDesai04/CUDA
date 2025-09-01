/*
Aim: Query and print whether the device can map host memory (`canMapHostMemory` property).

Thinking:
To accomplish this, I need a small CUDA C program that will query the properties of each CUDA-capable device present in the system and specifically check the `canMapHostMemory` field of the `cudaDeviceProp` structure. 
The steps are straightforward:
1. Include the necessary headers: <cuda_runtime.h> for CUDA runtime API and <cstdio> for printing.
2. In `main`, first call `cudaGetDeviceCount` to find how many devices are available.
3. If no devices are found, print a message and exit.
4. Iterate over each device index:
   a. Call `cudaGetDeviceProperties` to fill a `cudaDeviceProp` struct.
   b. Extract the `canMapHostMemory` boolean.
   c. Print the device index and whether the capability is true or false.
5. Handle any CUDA API errors by printing the error string using `cudaGetErrorString`.
6. Return 0 at the end.

This program is simple enough that a single file (.cu) is sufficient, and it will compile with `nvcc`. It directly addresses the requirement to query and print the `canMapHostMemory` property for each device.

*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        printf("No CUDA-capable devices detected.\n");
        return EXIT_SUCCESS;
    }

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, dev);
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to get properties for device %d: %s\n", dev, cudaGetErrorString(err));
            continue; // Skip to next device
        }

        printf("Device %d (%s): canMapHostMemory = %s\n",
               dev,
               prop.name,
               prop.canMapHostMemory ? "Yes" : "No");
    }

    return EXIT_SUCCESS;
}
