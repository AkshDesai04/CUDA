/*
Write a program to determine and print the 'warp size' for the default CUDA device. The warp size is a fundamental property that dictates how threads are scheduled. Find this value in the device properties struct.

Thinking:
- The goal is straightforward: query the CUDA device properties and output the warpSize field.
- We'll target the default device, which is typically device 0 after cudaSetDevice(0).
- Steps:
  1. Include <cuda_runtime.h> for CUDA runtime API.
  2. In main(), we call cudaGetDeviceCount to ensure at least one device is present.
  3. Use cudaGetDeviceProperties to obtain a cudaDeviceProp struct for device 0.
  4. Print the warpSize field.
- Handle error checks: if device count is zero or any CUDA call fails, print an error and exit.
- Keep the program simple: single thread on host prints the value.
- The code will compile with nvcc and produce an executable that prints something like "Warp size: 32".

Note: This code is self-contained and does not rely on any external dependencies beyond the CUDA runtime.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

int main(void)
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device count: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        return EXIT_FAILURE;
    }

    int device = 0; // default device
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", device, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    printf("Warp size: %d\n", prop.warpSize);

    return EXIT_SUCCESS;
}
