/*
Aim of the program:
Calculate the size in bytes required for the allocation (`256 * sizeof(int)`) and use that variable in both `cudaMalloc` and `cudaMemcpy`.

Thinking:
- The goal is to demonstrate proper use of a size variable when allocating and copying memory in CUDA.
- Compute the size once: `size_t bytes = 256 * sizeof(int);`
- Allocate host memory of that size and initialize it.
- Allocate device memory with `cudaMalloc(&d_arr, bytes);`
- Copy host to device using `cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);`
- Launch a simple kernel that, for example, increments each element.
- Copy the result back with `cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);`
- Print a few results to verify correctness.
- Include basic error checking for CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple kernel that increments each element by 1
__global__ void increment(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] += 1;
    }
}

int checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (%s): %s\n", msg, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

int main() {
    const int N = 256;
    size_t bytes = N * sizeof(int);  // Calculate size in bytes

    // Allocate and initialize host memory
    int *h_arr = (int *)malloc(bytes);
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }
    for (int i = 0; i < N; ++i) {
        h_arr[i] = i;  // initialize with 0..255
    }

    // Allocate device memory using the computed bytes
    int *d_arr;
    cudaMalloc((void **)&d_arr, bytes);
    if (checkCudaError("cudaMalloc")) {
        free(h_arr);
        return 1;
    }

    // Copy data from host to device using the same bytes variable
    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    if (checkCudaError("cudaMemcpy HostToDevice")) {
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Launch kernel: one thread per element
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    if (checkCudaError("Kernel launch")) {
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (checkCudaError("cudaMemcpy DeviceToHost")) {
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Print first 10 results to verify
    printf("First 10 elements after increment:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
