/*
Aim of the program: Add the `-arch=sm_XX` flag to your `nvcc` command line to compile for a specific compute capability (e.g., `sm_75`).

Thinking:
The user wants a CUDA program that demonstrates how to compile for a specific compute capability using the `-arch=sm_XX` flag. The program itself does not need to contain the flag; that flag is supplied during compilation. To illustrate usage, a simple vector addition kernel is sufficient. The code will allocate device memory, copy data, launch the kernel, copy the result back, and free resources. The comment at the top specifies the aim verbatim and includes my thought process on creating a minimal but functional CUDA example. The entire file is a valid `.cu` source file ready for compilation with a command such as:
    nvcc -arch=sm_75 my_program.cu -o my_program
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void vectorAdd(const float *A, const float *B, float *C, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

int main(void)
{
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_A[i] = i;
        h_B[i] = N - i;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("First 10 results: ");
    for (int i = 0; i < 10; ++i)
        printf("%f ", h_C[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
