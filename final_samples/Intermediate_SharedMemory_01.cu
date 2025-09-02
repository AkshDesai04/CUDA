```cuda
/*
Aim: Write a naive matrix multiplication kernel (C = A * B). Each thread will be responsible for calculating a single element of the output matrix C. This will involve the thread reading one full row from A and one full column from B from global memory.

Thinking:
- The goal is to implement the simplest possible CUDA kernel that performs matrix multiplication by assigning each thread to compute one element of the result matrix C.
- In a naive implementation, each thread reads an entire row of A and an entire column of B from global memory and accumulates the dot product. This leads to many global memory accesses and poor reuse of data, but it is straightforward to understand.
- For matrix dimensions, we use general M x K and K x N matrices. The thread indices map to the row (i) and column (j) of the result C. The kernel will loop over the inner dimension K to compute the sum.
- We choose a block size of 16x16 for simplicity. This is a typical small block size that fits well on many GPUs and gives enough parallelism for moderate matrix sizes. The grid dimensions are computed as (N+blockDim.x-1)/blockDim.x and (M+blockDim.y-1)/blockDim.y.
- Memory allocation and data transfer are performed on the host. We allocate device memory for A, B, and C, copy A and B from host to device, launch the kernel, then copy the result back.
- A simple reference CPU implementation is provided for correctness verification. We print a few elements of the result to confirm it matches the CPU version.
- Error checking macros are defined to catch CUDA API errors and kernel launch errors.
- Since this is a naive implementation, it does not use shared memory or tiling, so the performance is limited, but it serves as a clear example of basic matrix multiplication in CUDA.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Naive matrix multiplication kernel: each thread computes one element of C
__global__ void matMulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int K, int N)
{
    // Row index of C to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Column index of C to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Compute dot product of row of A and column of B
        for (int e = 0; e < K; ++e) {
            float a_elem = A[row * K + e];      // A[row][e]
            float b_elem = B[e * N + col];      // B[e][col]
            sum += a_elem * b_elem;
        }
        C[row * N + col] = sum;
    }
}

// Simple CPU implementation for verification
void matMulCPU(const float* A, const float* B, float* C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int e = 0; e < K; ++e) {
                sum += A[i * K + e] * B[e * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Utility to generate random matrix
void randomInit(float* mat, int size)
{
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)(rand()) / RAND_MAX;
    }
}

int main(void)
{
    // Matrix dimensions
    const int M = 512;   // rows of A, rows of C
    const int K = 512;   // columns of A, rows of B
    const int N = 512;   // columns of B, columns of C

    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_gpu = (float*)malloc(sizeC);
    float* h_C_cpu = (float*)malloc(sizeC);

    // Initialize matrices
    randomInit(h_A, M * K);
    randomInit(h_B, K * N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    matMulCPU(h_A, h_B, h_C_cpu, M, K, N);

    // Verify results (print a few elements)
    bool correct = true;
    for (int i = 0; i < M && correct; ++i) {
        for (int j = 0; j < N && correct; ++j) {
            float diff = fabs(h_C_gpu[i * N + j] - h_C_cpu[i * N + j]);
            if (diff > 1e-5) {
                printf("Mismatch at (%d,%d): GPU %f CPU %f\n",
                       i, j, h_C_gpu[i * N + j], h_C_cpu[i * N + j]);
                correct = false;
            }
        }
    }
    if (correct) {
        printf("Matrix multiplication is correct.\n");
    } else {
        printf("Matrix multiplication has errors.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```