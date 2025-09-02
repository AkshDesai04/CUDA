/*
Aim: Print the maximum number of resident blocks per multiprocessor.

Thinking:
The goal is to query the CUDA device properties and report the maximum number of resident blocks that can be scheduled on a single multiprocessor (SM) for a given GPU. The CUDA driver exposes this information in the cudaDeviceProp struct via the field `maxBlocksPerMultiprocessor`. We will:
1. Include the necessary headers (`stdio.h` for printing and `cuda_runtime.h` for CUDA runtime API).
2. In `main`, determine the current device (or the first device if no device is set).
3. Retrieve the device properties with `cudaGetDeviceProperties`.
4. Print out the `maxBlocksPerMultiprocessor` field.
5. Handle any CUDA API errors gracefully.
The program is self‑contained, compiles with `nvcc`, and outputs a single integer value representing the maximum resident blocks per SM.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
    int device = 0;
    cudaDeviceProp prop;
    cudaError_t err;

    // Get current device (if any). If none, default to device 0.
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        // If no device set, just use 0
        device = 0;
    }

    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum resident blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiprocessor);

    return 0;
}
