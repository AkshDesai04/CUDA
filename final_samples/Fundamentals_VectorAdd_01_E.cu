/*
 * Aim of the program: Implement vector subtraction: `C[i] = A[i] - B[i]`.
 *
 * [Thinking]
 * The program will perform element‑wise subtraction of two vectors A and B, storing the result in C.
 * A simple CUDA kernel is used: each thread handles one element. The kernel is launched with enough
 * blocks to cover the length of the vectors. Host memory is allocated for A, B and C. After initializing
 * A and B with test data, the arrays are copied to device memory. The kernel is launched, then the
 * resulting C array is copied back to the host. Finally, the results are printed for verification.
 * Basic error checking is included to ensure the kernel launch and memory operations succeed.
 * This example uses single‑precision floats and a fixed vector size, but it can easily be adapted
 * to larger sizes or to accept command‑line arguments.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define VECTOR_SIZE 1024  // Number of elements in each vector
#define THREADS_PER_BLOCK 256

// CUDA kernel for vector subtraction: C[i] = A[i] - B[i]
__global__ void vecSub(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] - B[idx];
    }
}

// Utility function to check for CUDA errors
void checkCudaError(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    int N = VECTOR_SIZE;
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;            // Example: A[i] = i
        h_B[i] = (float)(N - i);      // Example: B[i] = N - i
    }

    // Allocate device memory
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C");

    // Copy host memory to device
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Determine grid size
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vecSub<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // Verify results (simple print of first 10 elements)
    printf("First 10 results of C = A - B:\n");
    for (int i = 0; i < 10; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector subtraction completed successfully.\n");
    return 0;
}
