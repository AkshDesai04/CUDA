/*
Forget to call `cudaFree`. Run the program. This is a memory leak.

Thinking:
The task is to demonstrate a CUDA program that intentionally leaks device memory by allocating memory with cudaMalloc and never calling cudaFree. The aim is to show how forgetting to release GPU resources can lead to a memory leak. 
To keep the example simple, we will allocate an array of integers on the device, launch a trivial kernel that writes indices into that array, and then exit the program without freeing the memory. We avoid calling cudaDeviceReset to ensure the allocated memory remains unfreed at the end of the program. 
The kernel is minimal and serves only to illustrate that the memory is actually used. The program prints a message indicating that the kernel executed and that the memory was not freed. This satisfies the requirement of running the program and exhibiting a memory leak. 
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple kernel that writes thread indices into an array
__global__ void dummyKernel(int *d_arr, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        d_arr[idx] = idx;
    }
}

int main() {
    const int N = 1 << 20;           // 1,048,576 elements
    const size_t size = N * sizeof(int);
    int *d_arr = nullptr;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    // Launch kernel
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    dummyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        // Even if synchronization fails we intentionally skip freeing memory
        return 1;
    }

    std::cout << "Kernel executed. Device memory not freed (memory leak intentional)." << std::endl;

    // Intentionally do NOT call cudaFree(d_arr) to create a memory leak

    return 0;
}
