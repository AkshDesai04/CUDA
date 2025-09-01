/*
Implement for `double` data type.

Thinking:
The user asked only to "Implement for `double` data type." Without specifying a particular operation, I interpret this as providing a minimal CUDA program that demonstrates how to work with the `double` type in a GPU kernel. A classic example is element‑wise vector addition. This program will:
1. Allocate two input vectors and an output vector on the host.
2. Allocate device memory for these vectors.
3. Copy the inputs to the device.
4. Launch a kernel that adds corresponding elements using `double`.
5. Copy the result back to the host.
6. Print the first few results for verification.
The code is written in C/C++ CUDA (.cu file) and is fully self‑contained. It includes error checking for CUDA API calls. No additional text or files are produced, only the .cu source code as requested. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 256           // Number of elements in each vector
#define THREADS_PER_BLOCK 256

// CUDA kernel for element‑wise addition of two double arrays
__global__ void vectorAdd(const double* a, const double* b, double* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// Helper macro for checking CUDA errors
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main(void)
{
    double *h_a = NULL, *h_b = NULL, *h_c = NULL;
    double *d_a = NULL, *d_b = NULL, *d_c = NULL;
    size_t bytes = N * sizeof(double);

    /* Allocate host memory */
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (double)i;
        h_b[i] = (double)(2 * i);
    }

    /* Allocate device memory */
    CHECK_CUDA(cudaMalloc((void**)&d_a, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_b, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_c, bytes));

    /* Copy host data to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());  // Check for kernel launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Print a few results */
    printf("First 10 results of vector addition (double):\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("h_c[%d] = %f\n", i, h_c[i]);
    }

    /* Clean up */
    free(h_a);
    free(h_b);
    free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return EXIT_SUCCESS;
}
