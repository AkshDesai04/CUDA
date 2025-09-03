/*
Print the amount of shared memory available per multiprocessor.

Thought process:
- The goal is to query the CUDA device properties for the amount of shared memory available per multiprocessor.
- This information is stored in the `cudaDeviceProp.sharedMemPerMultiprocessor` field of the `cudaDeviceProp` structure.
- We will use `cudaGetDeviceProperties` to retrieve the properties for the default device (device 0).
- We'll then print this value in bytes and also convert it to megabytes for readability.
- Basic error checking will be added to ensure the CUDA API calls succeed.
- The program will be a simple console application that compiles with nvcc and prints the desired information.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    cudaDeviceProp prop;
    int device = 0; // default to device 0

    // Retrieve device properties
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Print shared memory per multiprocessor
    printf("Device %d (%s):\n", device, prop.name);
    printf("  Shared memory per multiprocessor: %zu bytes (%.2f MB)\n",
           prop.sharedMemPerMultiprocessor,
           prop.sharedMemPerMultiprocessor / (1024.0 * 1024.0));

    return 0;
}
