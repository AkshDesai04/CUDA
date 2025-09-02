```c
/*
 * Aim of the program: Ensure all host and device allocations use sizeof(double).
 *
 * Thinking:
 * The task is to write a CUDA C program that demonstrates correct memory
 * allocation for double precision data on both host and device.
 * A common mistake is to allocate memory using the element count directly
 * instead of multiplying by sizeof(double).  This program uses a simple
 * vector addition kernel to illustrate the proper use of sizeof(double)
 * in both host (malloc) and device (cudaMalloc) allocations.
 *
 * The program:
 *  - Defines a vector size N (1<<20 elements).
 *  - Allocates three host arrays (h_A, h_B, h_C) using malloc with
 *    N * sizeof(double).
 *  - Initializes h_A and h_B with some values.
 *  - Allocates three device arrays (d_A, d_B, d_C) with cudaMalloc,
 *    also using N * sizeof(double).
 *  - Copies h_A and h_B to d_A and d_B.
 *  - Launches a kernel that performs element‑wise addition: C = A + B.
 *  - Copies the result back to h_C.
 *  - Verifies a few elements for correctness.
 *  - Frees all allocated memory.
 *
 * By using sizeof(double) consistently, the program satisfies the
 * requirement and is safe across platforms with different double sizes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // 1M elements
#define BLOCK_SIZE 256

// CUDA kernel for vector addition
__global__ void vectorAdd(const double *A, const double *B, double *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void)
{
    double *h_A = NULL;
    double *h_B = NULL;
    double *h_C = NULL;

    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;

    size_t bytes = N * sizeof(double);

    // Allocate host memory using sizeof(double)
    h_A = (double *)malloc(bytes);
    h_B = (double *)malloc(bytes);
    h_C = (double *)malloc(bytes);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (double)i;
        h_B[i] = (double)(N - i);
    }

    // Allocate device memory using sizeof(double)
    cudaError_t err = cudaMalloc((void **)&d_A, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    err = cudaMalloc((void **)&d_B, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    err = cudaMalloc((void **)&d_C, bytes);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Copy host to device
    err = cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy A failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    err = cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy B failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Launch kernel
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy C failed: %s\n", cudaGetErrorString(err)); return EXIT_FAILURE; }

    // Verify results for a few elements
    int correct = 1;
    for (int i = 0; i < 10; ++i)
    {
        double expected = h_A[i] + h_B[i];
        if (h_C[i] != expected)
        {
            printf("Mismatch at index %d: host %f, device %f, expected %f\n",
                   i, h_C[i], expected, expected);
            correct = 0;
            break;
        }
    }
    if (correct)
    {
        printf("First 10 elements verified successfully.\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
```