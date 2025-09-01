/*
Aim of the program: Query and print the total number of registers available per thread block on the primary device.

Thinking:
- The CUDA Runtime API provides device attributes that expose hardware limits.
- One such attribute is `cudaDevAttrMaxRegistersPerBlock`, which reports the maximum number of 32‑bit registers that can be allocated to a single thread block on a given device.
- To retrieve this value, we use `cudaDeviceGetAttribute`. This function requires the attribute enum, a pointer to store the result, and the device ID.
- The "primary device" is typically device 0, but we query the current device to be safe. We can also set the device explicitly with `cudaSetDevice(0)`.
- After retrieving the value, we simply print it to standard output.
- Error handling: check the return status of CUDA API calls and report failures.

The program is written in CUDA C, compiled with `nvcc`, and contains no external dependencies beyond the CUDA Runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0; // Primary device
    cudaError_t err;

    // Ensure we are using the correct device
    err = cudaSetDevice(device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error setting device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // Retrieve the maximum number of registers per block
    int maxRegsPerBlock = 0;
    err = cudaDeviceGetAttribute(&maxRegsPerBlock,
                                 cudaDevAttrMaxRegistersPerBlock,
                                 device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error retrieving cudaDevAttrMaxRegistersPerBlock: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Maximum registers per thread block on device %d: %d\n", device, maxRegsPerBlock);

    return 0;
}
