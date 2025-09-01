/*
Implement matrix subtraction and multiplication.

My thought process: I considered that the user asked for matrix subtraction and multiplication. In CUDA, the simplest approach is to write two kernels: one for element‑wise subtraction (C[i] = A[i] – B[i]) and one for matrix multiplication (C[i,j] = Σk A[i,k] * B[k,j]). I opted for the naive implementation for clarity: each thread computes one element of the result. For subtraction, we map one thread per element. For multiplication, we map one thread per element of the result matrix, computing the dot product over the shared dimension. I used float type, a block size of 256 threads, and a grid size computed to cover all elements. The host code allocates matrices, initializes them with simple values, copies to device, launches kernels, copies results back, and prints a few entries. I also included a simple error‑checking macro for CUDA API calls. This code is self‑contained and can be compiled with nvcc to produce an executable that demonstrates both operations.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// Simple CUDA error checking macro
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            std::cerr << "CUDA error in " << __FILE__ << ":"      \
                      << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

// Kernel for matrix subtraction: C = A - B
__global__ void matrixSubKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total)
    {
        C[idx] = A[idx] - B[idx];
    }
}

// Kernel for matrix multiplication: C = A * B
__global__ void matrixMulKernel(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (idx < total)
    {
        int row = idx / N;
        int col = idx % N;
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
        {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[idx] = sum;
    }
}

int main()
{
    const int N = 512;                     // Matrix dimension (N x N)
    const int SIZE = N * N;
    const size_t bytes = SIZE * sizeof(float);

    // Allocate host memory
    float *h_A = new float[SIZE];
    float *h_B = new float[SIZE];
    float *h_C = new float[SIZE]; // Result for subtraction
    float *h_D = new float[SIZE];
    float *h_E = new float[SIZE];
    float *h_F = new float[SIZE]; // Result for multiplication

    // Initialize matrices with sample values
    for (int i = 0; i < SIZE; ++i)
    {
        h_A[i] = static_cast<float>(i % N);
        h_B[i] = static_cast<float>((i + 1) % N);
        h_D[i] = static_cast<float>((i + 2) % N);
        h_E[i] = static_cast<float>((i + 3) % N);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    float *d_D, *d_E, *d_F;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_E, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_F, bytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_E, h_E, bytes, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int THREADS_PER_BLOCK = 256;
    const int BLOCKS = (SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch subtraction kernel
    matrixSubKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch multiplication kernel
    matrixMulKernel<<<BLOCKS, THREADS_PER_BLOCK>>>(d_D, d_E, d_F, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_F, d_F, bytes, cudaMemcpyDeviceToHost));

    // Print a few elements of the results for verification
    std::cout << "First 5 elements of subtraction result (C = A - B):" << std::endl;
    for (int i = 0; i < 5; ++i)
        std::cout << h_C[i] << " ";
    std::cout << std::endl;

    std::cout << "First 5 elements of multiplication result (F = D * E):" << std::endl;
    for (int i = 0; i < 5; ++i)
        std::cout << h_F[i] << " ";
    std::cout << std::endl;

    // Clean up
    delete[] h_A; delete[] h_B; delete[] h_C;
    delete[] h_D; delete[] h_E; delete[] h_F;
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D)); CUDA_CHECK(cudaFree(d_E)); CUDA_CHECK(cudaFree(d_F));

    return 0;
}
