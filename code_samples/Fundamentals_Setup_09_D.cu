/*
Aim: Write a conditional print statement: if `concurrentKernels` is true, print 'Device supports concurrency.', otherwise print 'Device does not support concurrency.'

Thinking: 
To determine if a CUDA-capable device supports concurrent kernel execution, we query the device's properties using `cudaGetDeviceProperties`. The `cudaDeviceProp` structure contains a `concurrentKernels` boolean field. The program will:
1. Obtain the current device ID.
2. Retrieve the device properties for that ID.
3. Inspect `concurrentKernels`.
4. Print the appropriate message based on its value.
A simple error check for CUDA calls is included to ensure the program exits gracefully if any query fails. The code is kept minimal, focusing solely on the conditional print requirement.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get current device: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.concurrentKernels) {
        printf("Device supports concurrency.\n");
    } else {
        printf("Device does not support concurrency.\n");
    }

    return 0;
}
