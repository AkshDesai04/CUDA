```cuda
/* 
Aim of the program:
Write a kernel that performs a conditional operation. For two input vectors A and B, compute `C[i] = (A[i] > B[i]) ? A[i] : B[i]` (element-wise maximum).

Thinking:
To implement the requested element-wise maximum, we design a CUDA kernel that processes each element of the input vectors A and B in parallel. Each thread will compute one element of the output vector C by performing the conditional comparison `(A[i] > B[i]) ? A[i] : B[i]`. The kernel will be launched with enough threads to cover the entire input size, using a block size of 256 threads and a grid size computed as `(N + blockSize - 1) / blockSize` to handle cases where N is not a multiple of the block size.

The host code will:
1. Allocate host memory for vectors A, B, and C.
2. Initialize A and B with sample data (e.g., sequential numbers or random values).
3. Allocate device memory for A_d, B_d, and C_d.
4. Copy A and B from host to device.
5. Launch the kernel.
6. Copy the result C back from device to host.
7. Verify the result by comparing with a CPU implementation or by printing a few elements.
8. Clean up all allocated memory.

We include a simple error-checking macro to simplify CUDA API calls and ensure we capture any runtime errors. The program uses floating-point values for the vectors, but the logic is identical for integer types. The main function demonstrates usage and prints the first 10 elements of the result vector for quick sanity checking. */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

/* Kernel that computes element-wise maximum of two vectors */
__global__ void vecMaxKernel(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = (A[idx] > B[idx]) ? A[idx] : B[idx];
    }
}

int main(void)
{
    const size_t N = 1 << 20;          // Example: 1 million elements
    const size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays with sample data */
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)(i % 100);                 // Pattern: 0,1,2,...,99,0,1,...
        h_B[i] = (float)((100 - (i % 100)) % 100); // Complementary pattern
    }

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    vecMaxKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Simple verification: print first 10 elements */
    printf("First 10 elements of result vector C:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```