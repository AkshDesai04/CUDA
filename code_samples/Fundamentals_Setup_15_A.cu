/*
Write a program that allocates memory on the GPU, then calls `cudaDeviceReset()`, and then attempts to use or free that memory. Observe the error that occurs.

I want to demonstrate that after resetting the device with cudaDeviceReset, all previously allocated device memory becomes invalid. This invalidates pointers and the device context. Attempting to free or use such memory should result in CUDA errors. The code below allocates an array on the GPU, performs a device reset, then tries to free that memory and launch a kernel that writes to it. Error checking is performed to capture and display the CUDA error messages, illustrating the failure. The program is written in C for CUDA (.cu file) and includes a helper macro for concise error handling. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",             \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                      \
        }                                                             \
    } while (0)

__global__ void fillKernel(int *arr, int val)
{
    int idx = threadIdx.x;
    arr[idx] = val;
}

int main(void)
{
    int *devPtr = NULL;
    size_t size = 10 * sizeof(int);

    // Allocate memory on the GPU
    CUDA_CHECK(cudaMalloc((void **)&devPtr, size));
    printf("Allocated %zu bytes on device at pointer %p.\n", size, devPtr);

    // Reset the device (this invalidates all prior allocations and the context)
    CUDA_CHECK(cudaDeviceReset());
    printf("Device reset performed.\n");

    // Attempt to free the previously allocated memory
    cudaError_t err = cudaFree(devPtr);
    if (err != cudaSuccess) {
        printf("cudaFree after cudaDeviceReset returned error: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("cudaFree succeeded unexpectedly.\n");
    }

    // Attempt to launch a kernel that writes to the invalid memory
    fillKernel<<<1, 10>>>(devPtr, 42);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch after cudaDeviceReset returned error: %s\n",
               cudaGetErrorString(err));
    } else {
        printf("Kernel launch succeeded unexpectedly.\n");
    }

    return 0;
}
