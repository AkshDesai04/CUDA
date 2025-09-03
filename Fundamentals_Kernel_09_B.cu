/*
Create a __device__ function void increment(int* x) that increments the value pointed to by x. The kernel should call this on an element of an array in device memory.

Thinking:
1. We need a __device__ function that takes an int pointer and increments the integer it points to.
2. A kernel will receive a pointer to an array in device memory and an index; it will call increment on the element at that index.
3. Host code will allocate an array, copy it to device, launch the kernel, copy back, and print results.
4. Keep the example minimal: use a single element array for clarity, but also demonstrate multiple elements with a loop.
5. Compile with nvcc to produce a CUDA binary.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Device function that increments the value pointed to by x
__device__ void increment(int* x) {
    (*x)++;
}

// Kernel that calls increment on one element of the array
__global__ void incrementKernel(int* d_arr, int idx) {
    increment(&d_arr[idx]);
}

int main() {
    const int N = 5;                  // Size of the array
    int h_arr[N];                     // Host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;                 // Initialize with 0,1,2,...
    }

    // Allocate device memory
    int* d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Copy host array to device
    cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: each thread increments one element
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    incrementKernel<<<gridSize, blockSize>>>(d_arr, 0); // Only increment first element for demonstration

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy back to host
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Array after incrementing first element:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
