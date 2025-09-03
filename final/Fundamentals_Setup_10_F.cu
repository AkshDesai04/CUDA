/*
Aim: Query and print the memory bus width in bits.

Thinking:
The goal is to retrieve the memory bus width of the CUDA-capable device and display it.
CUDA provides a struct `cudaDeviceProp` that contains many device properties, one of which is `memoryBusWidth`.
We will:
1. Include the necessary headers: `<cuda_runtime.h>` for CUDA runtime API and `<stdio.h>` for I/O.
2. In `main`, call `cudaGetDeviceCount` to ensure at least one device is present.
3. Use `cudaGetDeviceProperties` to obtain the properties of device 0.
4. Extract `memoryBusWidth` from the `cudaDeviceProp` struct.
5. Print the bus width in bits to the console.
6. Handle any errors by printing descriptive messages and returning non‑zero exit codes.
The program is minimal, portable, and ready to compile with `nvcc`.
*/

#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to get device count: %s\n", cudaGetErrorString(err));
        return -1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found.\n");
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Failed to get device properties: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    return 0;
}
