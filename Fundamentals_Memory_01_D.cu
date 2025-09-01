/*
Instead of `malloc`, use `new int[256]` for the host allocation.
This program demonstrates a simple CUDA workflow where host memory is allocated
using C++'s `new` operator instead of the traditional C `malloc`. The host
array of 256 integers is initialized, copied to the device, processed by a
kernel that increments each element, then copied back to the host. Finally,
the program prints the first ten results and cleans up all allocated memory.
By using `new`, we illustrate how C++ memory allocation can be integrated
directly into CUDA host code while still leveraging device memory
manipulation.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256

// Kernel that increments each element of the array by 1
__global__ void increment(int *d_arr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] += 1;
    }
}

int main() {
    // Allocate host memory using C++ new operator
    int *h_arr = new int[N];
    if (h_arr == nullptr) {
        fprintf(stderr, "Host allocation failed!\n");
        return EXIT_FAILURE;
    }

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;
    }

    // Allocate device memory
    int *d_arr;
    cudaError_t err = cudaMalloc((void **)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Device allocation failed: %s\n", cudaGetErrorString(err));
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, N * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Launch kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocksPerGrid, threadsPerBlock>>>(d_arr);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Memcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        delete[] h_arr;
        return EXIT_FAILURE;
    }

    // Display first 10 elements
    printf("First 10 elements after increment:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    delete[] h_arr;

    return EXIT_SUCCESS;
}
