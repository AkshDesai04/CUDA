/*
Aim: Modify the program to launch with 32 blocks instead of 8.

Thinking:
The original instruction requested changing the kernel launch configuration from 8 blocks to 32 blocks. 
Since no existing code was provided, I created a minimal yet functional CUDA program that performs
element‑wise addition of two integer arrays.  The kernel is launched with a grid of 32 blocks
and each block contains 256 threads, yielding a total of 8192 threads which matches the size
of the arrays.  I included basic error checking, memory allocation, and a simple result
verification step.  The focus is on illustrating the block launch adjustment as requested.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (32 * 256)          // Number of elements
#define BLOCK_SIZE 256        // Threads per block
#define GRID_SIZE 32          // Number of blocks (modified from 8 to 32)

// CUDA kernel for vector addition
__global__ void vectorAdd(const int *A, const int *B, int *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                           \
    {                                                              \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                   \
        }                                                          \
    }

int main(void)
{
    int *h_A = (int*)malloc(N * sizeof(int));
    int *h_B = (int*)malloc(N * sizeof(int));
    int *h_C = (int*)malloc(N * sizeof(int));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = N - i;
    }

    int *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, N * sizeof(int)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice));

    // Launch kernel with GRID_SIZE blocks (32) and BLOCK_SIZE threads per block
    vectorAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Simple verification: print first 10 results
    printf("First 10 results of vector addition:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
