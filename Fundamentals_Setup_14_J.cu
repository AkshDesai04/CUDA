/*
Aim of the program: Query and print the maximum pitch in bytes for a memory allocation.

Thinking:
1. The maximum pitch is stored in the `cudaDeviceProp::memPitch` field.
2. We need to get device properties using `cudaGetDeviceProperties`.
3. Choose the current device (device 0 by default) or query `cudaGetDevice` first.
4. Handle any CUDA errors that might occur.
5. Print the result in a human-readable format.
6. The program is simple and self-contained, suitable for compiling with nvcc.

The program includes necessary headers, performs error checking,
and prints the maximum pitch in bytes.

*/

#include <stdio.h>
#include <cuda_runtime.h>

int main(void) {
    int device = 0;
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

    printf("Maximum pitch on device %d (%s): %zu bytes\n",
           device, prop.name, prop.memPitch);

    return 0;
}
