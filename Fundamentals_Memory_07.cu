/*
Aim: Write a program that tries to access a device pointer from host code (e.g., int* d_ptr; cudaMalloc(&d_ptr, ...); printf("%d", d_ptr[0]);). Observe the segmentation fault or error this causes and explain why it happens.

Thinking: In CUDA, cudaMalloc allocates memory on the GPU and returns a device pointer. The pointer value is an address that is only valid in the device address space. Host code runs in CPU memory space and cannot dereference a device address. Attempting to access d_ptr[0] from host will treat the device address as a host virtual address, leading to an invalid memory access, usually a segmentation fault. The correct way is to copy the data back to host with cudaMemcpy. The code below demonstrates this failure: it allocates an array on device, launches a kernel to write a value, then attempts to print the first element directly from host, which causes a crash. The program prints any CUDA errors before the crash for illustration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void init_array(int *a) {
    a[0] = 42;
}

int main() {
    int *d_ptr;
    cudaError_t err;

    // Allocate memory on the device
    err = cudaMalloc(&d_ptr, sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel to write to device memory
    init_array<<<1,1>>>(d_ptr);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    printf("Attempting to read device pointer directly from host...\n");
    // This line dereferences a device address from host code and will
    // usually cause a segmentation fault or other invalid memory access.
    printf("d_ptr[0] = %d\n", d_ptr[0]); // <-- invalid host access

    // Correct approach (for reference):
    // int h_val;
    // cudaMemcpy(&h_val, d_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    // printf("Correctly copied value: %d\n", h_val);

    cudaFree(d_ptr);
    return 0;
}
