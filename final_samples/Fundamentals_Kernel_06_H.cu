/*
Aim of the program: Allocate memory, but forget to copy the pointer's value back from the device. Print the host variable. What does it show and why?

Thinking: In CUDA, cudaMalloc writes the device address into the host variable passed to it. If we allocate memory on the device and then try to use that pointer on the host without copying it back (or forgetting to assign it), the host variable remains unchanged. In this simple example we allocate memory on the device, but we do not copy the resulting device pointer into another host pointer variable. We then print that host pointer. Since it was never assigned a value, it remains NULL (or whatever its initial value was). Therefore the program will print 0x0. This demonstrates that the host has no knowledge of the device address unless we explicitly store it in a host variable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int *devicePtr = NULL;   // Pointer to device memory
    int *hostPtr   = NULL;   // Host variable intended to receive the device address

    cudaError_t err = cudaMalloc((void**)&devicePtr, sizeof(int));
    if (err != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Normally we would copy the device address into hostPtr here,
    // but we intentionally skip that step to illustrate the point.

    printf("Host pointer variable after allocation: %p\n", (void*)hostPtr);

    // Clean up
    cudaFree(devicePtr);
    return 0;
}
