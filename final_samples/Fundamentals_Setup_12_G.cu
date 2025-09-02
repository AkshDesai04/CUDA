/*
Pass a host pointer (e.g., from `malloc`) directly to a kernel that expects a device pointer.

Thinking:
The goal of this program is to demonstrate that passing a host memory pointer directly to a CUDA kernel is incorrect and leads to undefined behavior. The kernel expects device memory addresses, so providing it a host pointer obtained via `malloc` will cause the kernel to read/write memory that it cannot access on the GPU, typically resulting in a segmentation fault or garbage results. 

To illustrate this, we allocate two arrays on the host, pass their pointers directly to a simple addition kernel, and then attempt to copy the result back from the GPU. The kernel will try to read from the host memory addresses as if they were device memory, which is invalid. We also provide the correct approach in comments for reference: allocate device memory with `cudaMalloc`, copy the data from host to device with `cudaMemcpy`, launch the kernel with device pointers, and then copy the result back.

This code intentionally includes the flawed usage to show the error, and the correct usage is commented out below. In a real debugging scenario, running this program would result in a runtime error or incorrect output. */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void addKernel(int *a, int *b, int *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int N = 8;
    const size_t bytes = N * sizeof(int);

    /* Allocate host memory */
    int *h_a = (int *)malloc(bytes);
    int *h_b = (int *)malloc(bytes);
    int *h_c = (int *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    /* ==============================
       Incorrect usage: passing host
       pointers directly to kernel
       ==============================
    */

    /* Launch kernel with host pointers (incorrect) */
    int threadsPerBlock = 4;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocks, threadsPerBlock>>>(h_a, h_b, h_c, N);

    /* Wait for GPU to finish */
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        // Even if kernel launch reports success, the memory access will be invalid.
    }

    /* Since h_c was never copied to device memory, its contents are undefined.
       Attempt to print results (likely garbage). */
    printf("Results (may be garbage or cause crash):\n");
    for (int i = 0; i < N; ++i) {
        printf("h_c[%d] = %d\n", i, h_c[i]);
    }

    /* ==============================
       Correct usage (for reference)
       ==============================
    */
    /*
    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    addKernel<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Correct results:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_c[%d] = %d\n", i, h_c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    */

    /* Cleanup host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
