/*
Aim of the program: Implement `C[i] = A[i] - scalar`

Thinking process:
1. Decide on a data type. I choose float for simplicity and to demonstrate typical CUDA usage.
2. Pick a reasonable array size. I use N = 1024, which is large enough to illustrate parallelism but small enough for easy debugging.
3. The operation is a simple element‑wise subtraction, so each thread will handle one element:
   C[i] = A[i] - scalar.
4. I will write a CUDA kernel `vecSubScalarKernel` that takes device pointers to A and C, the scalar, and the number of elements.
5. In the host code:
   - Allocate host arrays A_h and C_h.
   - Initialize A_h with some values (e.g., A_h[i] = i * 1.0f).
   - Define a scalar value, e.g., scalar = 5.0f.
   - Allocate device memory for A_d and C_d.
   - Copy A_h to A_d.
   - Determine grid and block dimensions (e.g., blockSize = 256, gridSize = (N + blockSize - 1)/blockSize).
   - Launch the kernel.
   - Copy C_d back to C_h.
   - Print a few results to verify correctness.
   - Clean up all allocated memory.
6. Add a simple error‑checking macro to wrap CUDA API calls.
7. Use `cudaFree` and `cudaFreeHost` (or `free`) appropriately.
8. Make sure to include necessary headers and compile with nvcc.

This program will be self‑contained and compile as a .cu file.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 256

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel to compute C[i] = A[i] - scalar
__global__ void vecSubScalarKernel(const float *A, float *C, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - scalar;
    }
}

int main(void) {
    float *A_h = (float*)malloc(N * sizeof(float));
    float *C_h = (float*)malloc(N * sizeof(float));
    if (!A_h || !C_h) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host array A_h with some values
    for (int i = 0; i < N; ++i) {
        A_h[i] = (float)i * 1.0f;  // e.g., 0.0, 1.0, 2.0, ...
    }

    float scalar = 5.0f;

    // Device pointers
    float *A_d = NULL;
    float *C_d = NULL;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&A_d, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&C_d, N * sizeof(float)));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vecSubScalarKernel<<<gridSize, BLOCK_SIZE>>>(A_d, C_d, scalar, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("First 10 results of C[i] = A[i] - %.1f:\n", scalar);
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("C[%d] = %f\n", i, C_h[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(C_d));
    free(A_h);
    free(C_h);

    // Successful exit
    return EXIT_SUCCESS;
}
