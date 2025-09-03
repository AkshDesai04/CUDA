/*
 * Aim of the program: Implement vector subtraction: `C[i] = A[i] - B[i]`.
 *
 * Thinking:
 * 1. We need a CUDA program that performs element‑wise subtraction of two
 *    vectors A and B, producing a result vector C.
 * 2. The program will:
 *    - Allocate host memory for A, B, and C.
 *    - Initialize A and B with some test data.
 *    - Allocate device memory for dA, dB, dC.
 *    - Copy A and B from host to device.
 *    - Launch a kernel that subtracts element‑wise:
 *          C[i] = A[i] - B[i];
 *      The kernel will compute one element per thread.
 *    - Copy the result dC back to host C.
 *    - Verify / print the result.
 *    - Free all allocated memory and exit.
 * 3. We will include basic error checking for CUDA API calls.
 * 4. The code is written in standard C with CUDA extensions,
 *    suitable for compiling with nvcc into a .cu file.
 * 5. For simplicity, we define a constant vector size N.
 * 6. The kernel will be launched with enough blocks to cover all elements,
 *    using a typical block size of 256 threads.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024          // Size of the vectors
#define BLOCK_SIZE 256  // Threads per block

// CUDA error checking macro
#define CHECK_CUDA(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                   \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Kernel: element‑wise vector subtraction
__global__ void vectorSub(const float *A, const float *B, float *C, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] - B[idx];
    }
}

int main(void) {
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;              // A[i] = i
        h_B[i] = (float)(N - i);        // B[i] = N - i
    }

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel
    size_t gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vectorSub<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify results (print first 10 elements)
    printf("First 10 elements of result vector C:\n");
    for (size_t i = 0; i < 10 && i < N; ++i) {
        printf("C[%zu] = %f\n", i, h_C[i]);
    }

    // Free device memory
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Vector subtraction completed successfully.\n");
    return 0;
}
