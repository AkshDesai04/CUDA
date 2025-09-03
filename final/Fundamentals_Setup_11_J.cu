/*
Create a simple Makefile to automate the compilation process.

Thinking:
The user asked for a simple Makefile, but the developer instruction explicitly requires producing a .cu file containing the entire code. Therefore, I must comply with the developer's higher priority instruction. I will embed the user's request verbatim in the comment, and then write a minimal CUDA program that can be compiled with a Makefile. This program will perform a simple vector addition to serve as an example of a CUDA kernel that can be built and run using a Makefile. The code is kept straightforward for clarity and to illustrate how a Makefile could compile and execute it.
*/

#include <stdio.h>
#include <stdlib.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = i * 0.001f;
        h_B[i] = i * 0.002f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Simple verification
    int correct = 1;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5f) {
            correct = 0;
            printf("Mismatch at %d: %f + %f != %f\n", i, h_A[i], h_B[i], h_C[i]);
            break;
        }
    }
    if (correct)
        printf("Vector addition succeeded!\n");

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
