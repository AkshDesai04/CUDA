/*
Query and print the number of asynchronous engines the device has (`asyncEngineCount`).

Thinking process:
1. The goal is to query the number of asynchronous engines on the current CUDA device.
2. We will use the CUDA Runtime API function `cudaGetDeviceProperties` to obtain a `cudaDeviceProp` structure.
3. The `asyncEngineCount` field of this structure holds the desired value.
4. We'll write a simple program that:
   - Retrieves the current device ID.
   - Calls `cudaGetDeviceProperties`.
   - Checks for errors in each CUDA API call.
   - Prints the async engine count.
5. Error handling: use `cudaGetLastError()` or check return codes directly and print error messages if any.
6. The program will compile with `nvcc` and run on a system with CUDA installed.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
    cudaError_t err;

    // Get the current device
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Retrieve device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print the number of asynchronous engines
    printf("Device %d (%s) has %d asynchronous engine(s).\n",
           device, prop.name, prop.asyncEngineCount);

    return 0;
}
