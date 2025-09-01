/*
Aim of the program: 
Write a conditional print statement: if `concurrentKernels` is true, print 'Device supports concurrency.', otherwise print 'Device does not support concurrency.'

Thought process:
1. The goal is to query a device property (concurrentKernels) using the CUDA runtime API.
2. We'll use `cudaGetDeviceProperties` to obtain a `cudaDeviceProp` struct for the current device (device 0 for simplicity).
3. The `concurrentKernels` field is a boolean that indicates if the device can execute multiple kernels concurrently.
4. In the main function, after retrieving the properties, we check this field:
   - If true, we print "Device supports concurrency."
   - Else, we print "Device does not support concurrency."
5. We'll include minimal error handling: if `cudaGetDeviceProperties` fails, we print an error and exit.
6. The program uses standard C (`stdio.h`) and CUDA runtime headers (`cuda_runtime.h`).
7. The code is self-contained, compiles with `nvcc`, and outputs the correct message when run.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int device = 0; // Use the first CUDA device
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (prop.concurrentKernels) {
        printf("Device supports concurrency.\n");
    } else {
        printf("Device does not support concurrency.\n");
    }

    return 0;
}
