/*
Create a `__device__` function that takes a pointer as an argument.

Thinking:
- The goal is to demonstrate a device function that accepts a pointer parameter.
- We'll create a simple `__device__` function `increment` that takes an `int*` and increments the value it points to.
- A kernel will call this function for each element in an array.
- Host code will allocate an array, copy it to the device, launch the kernel, copy the results back, and print them.
- Error checking will be minimal but sufficient to keep the example focused on the pointer function.
- This example serves both as a demonstration of pointer usage in a device function and as a complete, runnable CUDA program.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device function that takes a pointer as an argument and increments the value it points to.
__device__ void increment(int *ptr)
{
    // Simple operation: increment the integer at the given address.
    *ptr += 1;
}

// Kernel that calls the device function for each element of the array.
__global__ void kernel(int *dev_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Call the __device__ function with a pointer to the array element.
        increment(&dev_arr[idx]);
    }
}

int main(void)
{
    const int N = 10;
    int host_arr[N];

    // Initialize host array.
    for (int i = 0; i < N; ++i) {
        host_arr[i] = i;
    }

    // Allocate device memory.
    int *dev_arr = nullptr;
    cudaError_t err = cudaMalloc((void**)&dev_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device.
    err = cudaMemcpy(dev_arr, host_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        return 1;
    }

    // Launch kernel with sufficient threads.
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocks, threadsPerBlock>>>(dev_arr, N);

    // Wait for GPU to finish before accessing on host.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        return 1;
    }

    // Copy results back to host.
    err = cudaMemcpy(host_arr, dev_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        return 1;
    }

    // Print the results.
    printf("Array after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", host_arr[i]);
    }
    printf("\n");

    // Free device memory.
    cudaFree(dev_arr);

    return 0;
}
