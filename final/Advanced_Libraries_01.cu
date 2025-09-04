/*
Aim of the program (word to word as requested):

Perform a large matrix-matrix multiplication (SGEMM for floats) using the cuBLAS library. Initialize matrices A and B on the host, transfer them to the device, create a cuBLAS handle, and call the cublasSgemm function. Compare its performance to your custom kernel.

My entire thinking for this implementation:

1. The goal is to compare the performance of cuBLAS's highly optimized SGEMM against a straightforward custom kernel.  
2. I will choose square matrices of size 1024x1024 (M = N = K = 1024) which is large enough to show the performance difference but small enough to fit in the GPU memory of a typical device.  
3. Data layout: cuBLAS expects column-major matrices. For simplicity I will allocate the matrices in column-major order on the host and use the same layout on the device.  
4. The custom kernel will be a naive, non‑tiled implementation where each thread computes one element of the output matrix.  This keeps the code short and demonstrates the raw cost of lack of optimization.  
5. Timing will be done with CUDA events to obtain wall‑clock execution times.  Before and after each call we will record events and then compute the elapsed time.  
6. After each multiplication, I will copy the result back to the host and compute the maximum absolute difference between the cuBLAS result and the custom kernel result to confirm correctness.  
7. Error checking macros (`CUDA_CHECK`, `CUBLAS_CHECK`) will wrap CUDA and cuBLAS API calls to make debugging easier.  
8. The program will print the execution times for both implementations and the maximum difference found.  
9. I will include all required headers (`cuda_runtime.h`, `cublas_v2.h`, `<iostream>`, `<cstdlib>`) and compile with `nvcc`.  

This structure satisfies the prompt and provides a clear comparison between the library and a custom implementation.  
*/

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// Macro for checking CUDA runtime API errors
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error (" << cudaGetErrorString(err) << ") at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Macro for checking cuBLAS API errors
#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error (" << err << ") at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

// Naive SGEMM kernel: each thread computes one element of C
__global__ void sgemm_naive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K,
                            float alpha, float beta)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int p = 0; p < K; ++p) {
            // Column-major indexing: A[i + p*M], B[p + j*K]
            sum += A[row + p * M] * B[p + col * K];
        }
        // C[i + j*M] = alpha * sum + beta * C[i + j*M]
        C[row + col * M] = alpha * sum + beta * C[row + col * M];
    }
}

// Helper to compute maximum absolute difference between two arrays
float max_abs_diff(const float* a, const float* b, size_t size)
{
    float max_diff = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

int main()
{
    // Matrix dimensions
    const int M = 1024; // rows of A and C
    const int N = 1024; // columns of B and C
    const int K = 1024; // columns of A, rows of B

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C_cublas = new float[M * N];
    float* h_C_custom = new float[M * N];

    // Initialize matrices A and B with random values
    srand(0);
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C_cublas, size_C, cudaMemcpyHostToDevice)); // initialize C

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 1. cuBLAS SGEMM
    CUDA_CHECK(cudaEventRecord(start, 0));
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_A, M, // leading dimension of A is M
                             d_B, K, // leading dimension of B is K
                             &beta,
                             d_C, M)); // leading dimension of C is M
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_cublas_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_cublas_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C, size_C, cudaMemcpyDeviceToHost));

    // 2. Custom kernel SGEMM
    // Reinitialize C to zero before running custom kernel
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    CUDA_CHECK(cudaEventRecord(start, 0));
    sgemm_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float time_custom_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&time_custom_ms, start, stop));

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C_custom, d_C, size_C, cudaMemcpyDeviceToHost));

    // 3. Verify correctness
    float max_diff = max_abs_diff(h_C_cublas, h_C_custom, M * N);
    std::cout << "Maximum absolute difference between cuBLAS and custom: " << max_diff << std::endl;

    // 4. Print timings
    std::cout << "cuBLAS SGEMM time:   " << time_cublas_ms << " ms" << std::endl;
    std::cout << "Custom SGEMM time:   " << time_custom_ms << " ms" << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cublas;
    delete[] h_C_custom;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
