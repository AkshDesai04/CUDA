/*
Aim of the program: Create an error by calling a regular C++ function (that is not marked __device__) from within your kernel.

Thinking:
- The purpose is to demonstrate how the CUDA compiler enforces that only functions marked with __device__ (or __global__) can be called from device code.
- We define a simple regular C++ function, `hostFunction`, that prints a message. It is not decorated with any CUDA device qualifiers.
- The kernel `kernel` will attempt to call `hostFunction`. Since this function is not available on the device, the NVCC compiler will emit an error like "calling non-device function 'hostFunction' from device code".
- The main function launches the kernel and synchronizes, but compilation will fail before execution due to the illegal call.
- This file is intentionally designed to fail during compilation to illustrate the restriction.
*/

#include <stdio.h>

void hostFunction() {
    printf("Hello from host function!\n");
}

__global__ void kernel() {
    hostFunction(); // This should cause a compilation error
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
