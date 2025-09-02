```cuda
/*
Aim of the program: Verify the correctness of your naive matrix multiplication against a simple CPU implementation.

Thoughts:
The goal is to implement a straightforward (naive) matrix multiplication on the GPU and verify that the results match a reference CPU implementation. 
The CPU version uses three nested loops to compute the product of two N x N matrices A and B, producing C_cpu. 
On the GPU we launch a kernel where each thread computes one element of the output matrix C_gpgpu. 
Since the naive algorithm is simple, the kernel also loops over the shared dimension to accumulate the dot product. 
After running the kernel, we copy the result back to host memory and compare each element with the CPU result. 
If all elements match within a small tolerance, we print "Verification passed." Otherwise we report the first mismatch found. 
The code includes basic CUDA error checking, dynamic memory allocation, and a small test matrix size (e.g., N=512) that can be adjusted.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                                \
        }                                                                       \
    } while (0)

__global__ void matMulNaive(const float *A, const float *B, float *C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void initMatrix(float *M, int N)
{
    for (int i = 0; i < N * N; ++i) {
        M[i] = (float)(rand()) / RAND_MAX; // random between 0 and 1
    }
}

void cpuMatMul(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    const int N = 512; // matrix dimension (change as needed)
    const size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_cpu = (float*)malloc(bytes);
    float *h_C_gpu = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C_cpu || !h_C_gpu) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // CPU multiplication
    cpuMatMul(h_A, h_B, h_C_cpu, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    const float epsilon = 1e-4f;
    int correct = 1;
    for (int i = 0; i < N && correct; ++i) {
        for (int j = 0; j < N; ++j) {
            float cpuVal = h_C_cpu[i * N + j];
            float gpuVal = h_C_gpu[i * N + j];
            if (fabs(cpuVal - gpuVal) > epsilon) {
                printf("Mismatch at (%d, %d): CPU = %f, GPU = %f\n", i, j, cpuVal, gpuVal);
                correct = 0;
                break;
            }
        }
    }

    if (correct) {
        printf("Verification passed: GPU results match CPU results.\n");
    } else {
        printf("Verification failed: GPU results differ from CPU results.\n");
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_cpu);
    free(h_C_gpu);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```