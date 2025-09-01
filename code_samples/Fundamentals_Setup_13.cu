/*
Write a program that queries and prints the clock rate of the primary GPU in kilohertz (kHz).

Thinking process:

1. The goal is to use CUDA Runtime API to retrieve the clock rate of the primary GPU.
2. The primary GPU is usually device 0; we can use cudaGetDevice to confirm current device or simply use 0.
3. Use `cudaGetDeviceProperties` to fill a `cudaDeviceProp` structure.
4. The `clockRate` member of `cudaDeviceProp` holds the clock rate in kHz.
5. Print the value using `printf` with a clear message.
6. Include basic error checking: verify the return status of `cudaGetDeviceProperties`.
7. Compile with `nvcc` and run on a system with CUDA installed.
*/

#include <cstdio>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;  // Primary GPU
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Primary GPU (%d) clock rate: %d kHz\n", device, prop.clockRate);
    return 0;
}
