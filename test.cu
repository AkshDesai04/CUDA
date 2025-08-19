#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1024  // Matrix size (1024x1024)

void generateRandomMatrix(float *matrix, int size) {
    for (int i = 0; i < size * size; ++i) {
        matrix[i] = (float)(rand() % 100);  // Random values between 0 and 99
    }
}

int main() {
    float *A, *B, *C;   // Host matrices
    float *d_A, *d_B, *d_C;   // Device matrices

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    A = (float *)malloc(size);
    B = (float *)malloc(size);
    C = (float *)malloc(size);

    // Generate random matrices
    generateRandomMatrix(A, N);
    generateRandomMatrix(B, N);

    // Allocate device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Initialize cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform matrix multiplication using cuBLAS (C = A * B)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);

    // Copy result back to host
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Print part of the result (optional)
    // for (int i = 0; i < 5; i++) {
    //     for (int j = 0; j < 5; j++) {
    //         printf("%f ", C[i * N + j]);
    //     }
    //     printf("\n");
    // }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);

    return 0;
}
