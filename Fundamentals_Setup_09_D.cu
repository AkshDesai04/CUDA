/*
Aim: Write a conditional print statement: if `concurrentKernels` is true, print 'Device supports concurrency.', otherwise print 'Device does not support concurrency.'

Thoughts:
To accomplish this, I need a simple CUDA host program that queries the properties of the current CUDA device using the CUDA Runtime API. The key property is `concurrentKernels`, which indicates whether the device supports launching multiple kernels concurrently. 
1. Include the necessary headers: `<cuda_runtime.h>` for CUDA API and `<stdio.h>` for printing.
2. In `main()`, determine the current device using `cudaGetDevice()`.
3. Retrieve the device properties with `cudaGetDeviceProperties()`.
4. Check `prop.concurrentKernels`. If it is non-zero, print the supportive message; otherwise print the negative message.
5. Return 0.

No device-side kernel is needed. The program is minimal and self-contained, ready to compile with `nvcc`. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // Get the current device ID
    int device = 0;
    cudaError_t err = cudaGetDevice(&device);
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

    // Conditional print based on concurrentKernels support
    if (prop.concurrentKernels) {
        printf("Device supports concurrency.\n");
    } else {
        printf("Device does not support concurrency.\n");
    }

    return 0;
}
