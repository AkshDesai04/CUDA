/*
Aim: Query and print the `asyncEngineCount` property, which indicates the number of engines available for overlapping data copies and kernel execution.

Thinking:
- The goal is to retrieve and display the asyncEngineCount of a CUDA-capable device.
- We'll use the CUDA runtime API, specifically cudaGetDeviceProperties, to obtain a cudaDeviceProp structure.
- The asyncEngineCount field in cudaDeviceProp gives the number of asynchronous engines.
- We'll query the properties of the current device (device 0 for simplicity).
- Error checking will be performed after calling CUDA runtime functions.
- The program will simply print the asyncEngineCount value to standard output.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    cudaError_t err;
    int device = 0;  // Use the first device (you can query device count if needed)

    // Get the device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the asyncEngineCount
    printf("Device %d (%s) asyncEngineCount: %d\n", device, prop.name, prop.asyncEngineCount);

    return 0;
}
