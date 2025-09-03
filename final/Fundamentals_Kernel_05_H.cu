/*
Pass two arrays, `in_a` and `in_b`, and one output array `out_c`. Have each thread with global ID `id` compute `out_c[id] = in_a[id] + in_b[id]`.

Goal: Write a CUDA C program that performs element‑wise addition of two input arrays.  
The design consists of a simple device kernel that reads from two constant‑input
pointers and writes the sum to a third pointer, using the global thread index
(`blockIdx.x * blockDim.x + threadIdx.x`) as the element index.  
The host code allocates host memory, initializes example data, copies the
arrays to device memory, launches the kernel with an appropriate grid/block
configuration, copies the result back, and prints a few sample values for
verification.  
Error checking macros are included for robust CUDA API calls. The program is
self‑contained, uses single‑precision floats, and can be compiled with
`nvcc` producing a .cu file. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void vecAdd(const float *in_a, const float *in_b, float *out_c, int N) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) {
        out_c[id] = in_a[id] + in_b[id];
    }
}

int main(void) {
    const int N = 1 << 20;  // Number of elements (1M)
    const size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CHECK_CUDA(cudaMalloc((void **)&d_a, size));
    CHECK_CUDA(cudaMalloc((void **)&d_b, size));
    CHECK_CUDA(cudaMalloc((void **)&d_c, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results (index: a + b = c):\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d: %f + %f = %f\n", i, h_a[i], h_b[i], h_c[i]);
    }
    printf("...\n");
    printf("%d: %f + %f = %f\n", N-1, h_a[N-1], h_b[N-1], h_c[N-1]);

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
