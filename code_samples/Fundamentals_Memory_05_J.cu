/*
Forget to call `cudaFree`. Run the program. This is a memory leak.

Thinking process:
The user wants a CUDA program that deliberately does not free GPU memory, to illustrate a memory leak scenario. The aim is to keep the code simple while clearly showing allocation, kernel launch, and data transfer without the cleanup step. The program will allocate an array on the device, perform a simple operation in a kernel, copy results back to host, print the result, and exit without calling cudaFree. This demonstrates that the allocated device memory remains until the process ends, causing a leak if the program is run repeatedly or in a long-lived application. The code will be self-contained and can be compiled with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element by 1
__global__ void incrementKernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int main(void) {
    const int N = 1024;
    size_t size = N * sizeof(int);

    // Host array
    int h_arr[N];
    for (int i = 0; i < N; ++i) h_arr[i] = i;

    // Device array pointer
    int *d_arr = NULL;

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel: 256 threads per block
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Print first 10 results for verification
    printf("First 10 results after kernel execution:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Intentionally omit cudaFree(d_arr) to create a memory leak

    // Clean exit
    return 0;
}
