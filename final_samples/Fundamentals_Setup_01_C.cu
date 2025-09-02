/*
Aim of the program: 
Write a program to explicitly set the active CUDA device to 0 using `cudaSetDevice`. Verify the call was successful by then calling `cudaGetDevice` and printing the result.

Thinking and design:
- We need a minimal CUDA C program that runs on the host and interacts with the CUDA runtime.
- The program will call `cudaSetDevice(0)` to set device 0 as the active device.
- Afterwards, we call `cudaGetDevice` to retrieve the currently active device.
- Both CUDA runtime API calls return a `cudaError_t`. We should check for errors and print any error messages using `cudaGetErrorString`.
- If everything succeeds, we print the retrieved device number (expected to be 0).
- The program should compile with `nvcc` as a .cu file. We'll include `<stdio.h>` for printing and `<cuda_runtime.h>` for CUDA runtime functions.
- The main function will perform the steps and exit with 0 on success, or non-zero if any CUDA error occurs.

The code is intentionally straightforward, focusing only on the requested functionality: setting device, getting device, verifying, and printing.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    // Set the active CUDA device to 0
    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaSetDevice(0): %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Get the currently active CUDA device
    int dev;
    err = cudaGetDevice(&dev);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in cudaGetDevice: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Print the result
    printf("Currently active CUDA device: %d\n", dev);

    return 0;
}
