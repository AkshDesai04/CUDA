```cuda
/*
Aim of the program: Implement with `double` precision.

Thought process:
The user has requested a CUDA program implemented in C (CUDA C) that uses double precision. The specific computation was not specified, so I chose a common and illustrative example: element-wise vector addition (A + B = C). This showcases how to use double precision (double type) on the GPU, transfer data between host and device, launch a kernel, and verify the result on the host. The program includes:

1. Inclusion of necessary headers (`stdio.h`, `stdlib.h`, `cuda_runtime.h`).
2. A CUDA kernel that performs the addition using double precision.
3. Host functions to allocate and initialize input vectors, invoke the kernel, and copy back the result.
4. Simple error checking for CUDA API calls.
5. Printing a few results to confirm correctness.
6. Proper cleanup of allocated memory.

This provides a complete, self-contained .cu file that compiles with `nvcc` and demonstrates double precision usage on the GPU. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024          // Number of elements in vectors
#define BLOCK_SIZE 256  // Threads per block

// CUDA kernel for element-wise addition of two double arrays
__global__ void vectorAdd(const double *A, const double *B, double *C, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

int main(void)
{
    double *h_A = NULL, *h_B = NULL, *h_C = NULL;
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t size = N * sizeof(double);

    // Allocate host memory
    h_A = (double *)malloc(size);
    h_B = (double *)malloc(size);
    h_C = (double *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = (double)i * 1.0;
        h_B[i] = (double)(N - i) * 1.0;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy input vectors from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorAdd<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result vector back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify result (print first 5 elements)
    printf("First 5 results of A + B:\n");
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```