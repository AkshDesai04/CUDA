/*
Modify the function to use `cudaMemset` to initialize all bytes to `0xFF` (which corresponds to -1 for signed integers).

The task is to show how to use `cudaMemset` to set device memory to all 0xFF bytes. 0xFF in two's‑complement represents –1 for signed 32‑bit integers, so after the call each `int` element in device memory will contain –1.  A simple kernel will then increment each element, turning the –1 values into 0.  The program demonstrates allocation, `cudaMemset`, kernel launch, memory copy back to host, and result printing.  Error handling is included to catch any CUDA runtime failures.  The code is self‑contained, fully compilable with `nvcc`, and showcases the required modification.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Helper to check CUDA errors
void checkCudaError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Simple kernel that increments each element
__global__ void incrementKernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1;
    }
}

int main() {
    const int N = 10;
    const size_t size = N * sizeof(int);

    int *devPtr = nullptr;
    cudaMalloc((void**)&devPtr, size);
    checkCudaError("cudaMalloc");

    // Initialize all bytes to 0xFF (i.e., each int becomes -1)
    cudaMemset(devPtr, 0xFF, size);
    checkCudaError("cudaMemset");

    // Launch kernel to increment each element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    incrementKernel<<<blocks, threadsPerBlock>>>(devPtr, N);
    checkCudaError("incrementKernel launch");

    // Copy result back to host
    int hData[N];
    cudaMemcpy(hData, devPtr, size, cudaMemcpyDeviceToHost);
    checkCudaError("cudaMemcpy");

    // Print results (should all be 0)
    for (int i = 0; i < N; ++i) {
        printf("hData[%d] = %d\n", i, hData[i]);
    }

    cudaFree(devPtr);
    return 0;
}