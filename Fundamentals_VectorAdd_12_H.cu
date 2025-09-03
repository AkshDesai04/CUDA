/*
Verify the matrix addition result on the CPU.
The goal of this program is to demonstrate a simple matrix addition performed on the GPU using CUDA, followed by a verification step executed on the CPU to ensure correctness of the result. 
The program follows these main steps:
1. Define the matrix dimensions and allocate host memory for matrices A, B, and C. 
2. Initialize matrices A and B with deterministic values so that the expected result (A + B) is known. 
3. Allocate device memory for d_A, d_B, and d_C and copy A and B from host to device. 
4. Launch a CUDA kernel that performs element-wise addition of the two input matrices and stores the result in d_C. 
5. Copy the result matrix d_C back from device to host into matrix C. 
6. On the CPU, iterate through all elements of C and compare each entry against the expected value computed as A[i][j] + B[i][j]. If any element differs, report the index and the incorrect values; otherwise, confirm that the verification passed. 
7. Clean up all allocated resources and exit. 
The program includes basic CUDA error checking and uses a simple 2D grid of thread blocks (block size 16x16) to cover the entire matrix. It is written in plain C/C++ with CUDA extensions, suitable for compilation with nvcc to produce a .cu file. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  // Matrix dimension (N x N)
#define BLOCK_SIZE 16

// CUDA kernel for matrix addition
__global__ void matrixAdd(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y-index
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x-index

    if (row < n && col < n)
    {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",             \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main()
{
    size_t bytes = N * N * sizeof(float);
    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices A and B
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(i % 100);
        h_B[i] = (float)((i * 2) % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matrixAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result on CPU
    int errors = 0;
    for (int i = 0; i < N * N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f) {
            if (errors < 10) { // print first few errors
                printf("Mismatch at index %d: GPU %f, Expected %f\n",
                       i, h_C[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed: all elements match.\n");
    } else {
        printf("Verification failed: %d mismatches found.\n", errors);
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return EXIT_SUCCESS;
}
