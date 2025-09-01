/*
Aim: Convert and print the clock rate in gigahertz (GHz).

Thought process:
- The CUDA runtime provides device properties via `cudaGetDeviceProperties`.
- The property `clockRate` is expressed in kilohertz (kHz). To convert to gigahertz (GHz) we divide by 1,000,000 (since 1 GHz = 1,000,000 kHz).
- We will query the first available GPU device (device 0) and print its clock rate in GHz with two decimal precision.
- Error checking is minimal but ensures the CUDA calls succeed. If a device cannot be found or the query fails, we will output an error message.
- The program is a simple host-only CUDA program; no kernels are launched. It is written in standard C/C++ syntax and compiled with `nvcc` as a `.cu` file.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return 1;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    // Use the first device (device 0)
    int device = 0;
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Convert clock rate from kHz to GHz
    double gHz = prop.clockRate / 1.0e6;

    printf("Device %d: %s\n", device, prop.name);
    printf("Clock rate: %.2f GHz\n", gHz);

    return 0;
}
