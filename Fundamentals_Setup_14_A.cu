```c
/*
A more direct property is `deviceOverlap`. Query and print this boolean property for the primary device.

Thinking: The goal is to create a minimal CUDA C program that queries the device properties for the primary GPU device (device 0) and prints the value of the `deviceOverlap` boolean flag. 
The `cudaGetDeviceProperties` function fills a `cudaDeviceProp` struct which contains a `int` field `deviceOverlap` that is non-zero if the device supports concurrent execution of copy and kernel operations. 
We will check for errors from `cudaGetDeviceProperties`, then print the boolean value in a human-readable form (`true` or `false`). 
No kernel launch or complex setup is required. The program will compile with `nvcc` and run on any system with a CUDA-capable device. 
We will use standard C I/O (`printf`) and include `<cuda_runtime.h>` for the CUDA runtime API.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // Primary device
    cudaDeviceProp prop;

    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("deviceOverlap: %s\n", prop.deviceOverlap ? "true" : "false");
    return EXIT_SUCCESS;
}
```