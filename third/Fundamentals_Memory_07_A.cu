/*
Aim: Explain in a comment the concept of separate host and device memory address spaces.
This program demonstrates the separation of host (CPU) and device (GPU) memory spaces in CUDA.
On the host side, we allocate a vector of integers in standard CPU memory.
On the device side, we allocate a corresponding vector in GPU global memory using cudaMalloc.
These two memory spaces are completely separate: addresses in host memory cannot be dereferenced
directly by device code, and addresses in device memory cannot be accessed directly by host code.
To transfer data between them, we must explicitly copy using cudaMemcpy.
In this example, we perform a simple vector addition: c = a + b.
The kernel runs on the GPU using the device memory addresses.
After the kernel completes, we copy the result back to host memory for printing.
The program shows the typical workflow: allocate host memory, allocate device memory,
copy data from host to device, launch kernel, copy results back, free memory.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const int *a, const int *b, int *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 10;
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_a = (int*)malloc(size);
    int *h_b = (int*)malloc(size);
    int *h_c = (int*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate device memory
    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    // Copy result from device back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Result:\n");
    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
