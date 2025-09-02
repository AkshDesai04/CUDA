/*
Time the multiplication kernel.

The goal of this program is to measure the execution time of a CUDA kernel that performs matrix multiplication.
To do this we use CUDA events, which provide high‑resolution timestamps on the GPU.  The workflow is:

1. Allocate two input matrices A and B and one output matrix C on the host.
2. Fill A and B with deterministic data so the result is reproducible.
3. Allocate corresponding memory on the device and copy the inputs over.
4. Create two CUDA events, one before the kernel launch and one after.
5. Launch a simple, naive matrix‑multiplication kernel with a 2‑D grid of thread blocks.
6. Record the events and compute the elapsed time in milliseconds.
7. Copy the result back to the host (not strictly needed for timing, but a realistic workflow).
8. Print the measured time, verify the correctness of the result (optional), and free all memory.

The kernel uses a standard row‑major layout.  Each thread computes one element of the result matrix by iterating over
the shared dimension.  For simplicity and clarity we ignore optimizations such as shared memory tiling.

Error checking is performed with a helper macro that prints a message and aborts on CUDA API failures.

This code can be compiled with nvcc as:

    nvcc -o matrix_mul_time matrix_mul_time.cu

and executed.  It will output the time taken by the kernel in milliseconds.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // Thread block dimensions (16x16)

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

// Naive matrix multiplication kernel: C = A * B
__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // M rows
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // N columns

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(void)
{
    // Matrix dimensions
    const int M = 1024;  // Rows of A and C
    const int K = 1024;  // Columns of A, rows of B
    const int N = 1024;  // Columns of B and C

    const size_t sizeA = M * K * sizeof(float);
    const size_t sizeB = K * N * sizeof(float);
    const size_t sizeC = M * N * sizeof(float);

    // Host allocation
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host matrices with deterministic values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;  // For simplicity, all ones
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = 1.0f;  // All ones
    }

    // Device allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&d_C, sizeC));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    printf("Matrix multiplication kernel time: %f ms\n", milliseconds);

    // Optional: copy result back to host (not necessary for timing)
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
