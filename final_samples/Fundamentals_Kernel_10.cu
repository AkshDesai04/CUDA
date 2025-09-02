/*
Aim of the program: Define a simple C struct on the host. Pass an instance of this struct by value to a kernel. The kernel should print one of the struct's members to verify it was received correctly.

Thinking:
- Define a plain C struct with a few simple members (int and float) so that it is trivially copyable.
- Allocate an instance on the host and initialize its fields.
- Write a kernel that accepts the struct by value; inside the kernel, use device-side printf to output one of the fields.
- Launch the kernel with a single thread (1 block, 1 thread) because the goal is just to demonstrate passing by value and printing.
- Since device printf is used, we must compile with a compute capability that supports it (>= 2.0). The code is written generically; the user can specify the architecture flag when compiling with nvcc.
- After launching the kernel, call cudaDeviceSynchronize to ensure all device output is flushed before the host program exits.
- Include necessary error checks for the kernel launch and CUDA calls for robustness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct Simple {
    int a;
    float b;
};

__global__ void printStructByValue(Simple s) {
    // Print one member to verify correct reception
    printf("Device: received struct with a = %d, b = %f\n", s.a, s.b);
}

int main(void) {
    // Host-side struct initialization
    Simple hostStruct;
    hostStruct.a = 42;
    hostStruct.b = 3.14159f;

    // Launch kernel with one thread
    printStructByValue<<<1, 1>>>(hostStruct);

    // Synchronize to wait for kernel completion and flush device printf
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error after kernel launch: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: print from host for completeness
    printf("Host: struct was sent to device with a = %d, b = %f\n", hostStruct.a, hostStruct.b);

    return 0;
}
