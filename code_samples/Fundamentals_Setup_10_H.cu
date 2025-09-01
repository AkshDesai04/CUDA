/*
Query and print the GPU's clock rate in GHz.

Thought Process:
1. The goal is to write a CUDA program in C that retrieves the GPU's clock rate and prints it in GHz.
2. CUDA provides the `cudaGetDeviceProperties` function which fills a `cudaDeviceProp` struct that contains the `clockRate` field in kHz.
3. The program will:
   - Include `<stdio.h>` for I/O and `<cuda_runtime.h>` for CUDA runtime functions.
   - Use `cudaGetDeviceCount` to confirm that at least one device exists.
   - Choose device 0 for simplicity.
   - Call `cudaGetDeviceProperties` for that device.
   - Convert the `clockRate` from kHz to GHz: `clockRate_GHz = clockRate_kHz / 1e6`.
   - Print the result with a reasonable precision.
4. Error handling will be performed by checking the return status of CUDA API calls.
5. The program will compile with `nvcc` and run on a system with a CUDA-capable GPU.

This file is a complete CUDA C source that can be compiled as a `.cu` file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void)
{
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

    int device = 0;  // Use the first device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get properties for device %d: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    // clockRate is in kHz; convert to GHz
    double clockGhz = prop.clockRate / 1e6;

    printf("GPU (%s) clock rate: %.3f GHz\n", prop.name, clockGhz);

    return 0;
}
