/*
Ensure all host and device allocations use `sizeof(double)`.

Thought process:
- The goal is to demonstrate that all memory allocations (both host side via malloc and device side via cudaMalloc) use sizeof(double).
- We'll create two input vectors a and b of size N, fill them, allocate device memory for a, b, c.
- We'll copy a and b to device, launch a simple vector add kernel that computes c[i] = a[i] + b[i].
- Then copy c back to host and print a few results to verify.
- We'll check error returns from cuda calls and report any issues.
- All allocations will use sizeof(double) * N.
- Use standard includes, define N=1024, threads per block = 256.
- Use proper error checking macro.
- Ensure code compiles as a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vecAdd(const double *a, const double *b, double *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1024;
    const size_t size = N * sizeof(double);

    // Host allocations
    double *h_a = (double*)malloc(size);
    double *h_b = (double*)malloc(size);
    double *h_c = (double*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_a[i] = (double)i;
        h_b[i] = (double)(N - i);
    }

    // Device allocations
    double *d_a = NULL;
    double *d_b = NULL;
    double *d_c = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_a, size));
    CUDA_CHECK(cudaMalloc((void**)&d_b, size));
    CUDA_CHECK(cudaMalloc((void**)&d_c, size));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    // Print first 5 results for verification
    printf("Index\tA\tB\tC=A+B\n");
    for (int i = 0; i < 5; ++i) {
        printf("%d\t%.1f\t%.1f\t%.1f\n", i, h_a[i], h_b[i], h_c[i]);
    }

    // Clean up
    free(h_a);
    free(h_b);
    free(h_c);
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
