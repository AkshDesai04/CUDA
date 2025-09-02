/*
Aim: Perform a large matrix-matrix multiplication (SGEMM for floats) using the cuBLAS library. Initialize matrices A and B on the host, transfer them to the device, create a cuBLAS handle, and call the cublasSgemm function. Compare its performance to your custom kernel.

Thinking:
1. We want to demonstrate both a custom CUDA kernel for matrix multiplication and the cuBLAS SGEMM routine, measuring execution times for comparison.
2. Matrix dimensions will be large (2048 x 2048) to make the GPU performance differences noticeable. We set M = N = K = 2048.
3. cuBLAS expects column‑major matrices. To keep memory access simple we will store all matrices in column‑major order. That means element (row, col) is stored at index col*M + row for an MxN matrix.
4. Host memory: allocate three float arrays: hA[M*K], hB[K*N], hC_cublas[M*N], hC_custom[M*N]. Fill hA and hB with random floats.
5. Device memory: allocate dA[M*K], dB[K*N], dC_cublas[M*N], dC_custom[M*N] with cudaMalloc. Transfer hA and hB to dA and dB via cudaMemcpy.
6. Custom kernel: a simple tiling‑free kernel where each thread computes one output element C[row][col] by summing over k. Block dimension 16x16 is chosen to keep occupancy high. The kernel uses column‑major indexing to match cuBLAS. Launch grid dimensions ceil(M/16) x ceil(N/16).
7. Timing: Use CUDA events for both kernels. Record start and stop events around the kernel launch and cuBLAS call. Use cudaEventElapsedTime to get milliseconds. Additionally, cudaDeviceSynchronize is called after each kernel to ensure completion before timing ends.
8. cuBLAS call: create a handle with cublasCreate. Call cublasSgemm with parameters: operation type CUBLAS_OP_N for both A and B, dimensions M, N, K, pointers dA, dB, dC_cublas, leading dimensions lda = M, ldb = K, ldc = M. The result is stored in dC_cublas.
9. After both computations, copy results back to host: cudaMemcpy hC_custom and hC_cublas. Optionally compare the two results to verify correctness (e.g., check the max difference).
10. Print execution times and maybe the time difference or speedup.
11. Clean up: destroy cuBLAS handle, free device memory, free host memory.
12. Error checking: define a macro CUDA_CHECK to wrap CUDA API calls and exit on failure. Similarly for cuBLAS calls.

The program is self‑contained and uses only standard CUDA and cuBLAS libraries. It demonstrates performance differences between a naive CUDA implementation and the highly optimised cuBLAS SGEMM routine. */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cmath>

#define CUDA_CHECK(err) do { \
    cudaError_t err__ = (err); \
    if (err__ != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(err) do { \
    cublasStatus_t err__ = (err); \
    if (err__ != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error: " << err__ << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Custom matrix multiplication kernel (Naive, each thread computes one element)
__global__ void matMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                             int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index

    if (row < M && col < N) {
        float sum = 0.0f;
        // Column-major indexing: element (r,c) -> index c*M + r
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + row] * B[col * K + k];
        }
        C[col * M + row] = sum;
    }
}

int main()
{
    // Matrix dimensions
    const int M = 2048; // rows of A and C
    const int K = 2048; // columns of A and rows of B
    const int N = 2048; // columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *hA = new float[M * K];
    float *hB = new float[K * N];
    float *hC_cublas = new float[M * N];
    float *hC_custom = new float[M * N];

    // Initialize A and B with random floats
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < M * K; ++i) hA[i] = dist(rng);
    for (int i = 0; i < K * N; ++i) hB[i] = dist(rng);

    // Allocate device memory
    float *dA, *dB, *dC_cublas, *dC_custom;
    CUDA_CHECK(cudaMalloc((void**)&dA, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&dB, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&dC_cublas, sizeC));
    CUDA_CHECK(cudaMalloc((void**)&dC_custom, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up cuBLAS
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             dA, M,
                             dB, K,
                             &beta,
                             dC_cublas, M));

    // Timing cuBLAS SGEMM
    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             dA, M,
                             dB, K,
                             &beta,
                             dC_cublas, M));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_cublas = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, start, stop));

    // Timing custom kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

    CUDA_CHECK(cudaEventRecord(start));
    matMulKernel<<<gridDim, blockDim>>>(dA, dB, dC_custom, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_custom = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_custom, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hC_cublas, dC_cublas, sizeC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hC_custom, dC_custom, sizeC, cudaMemcpyDeviceToHost));

    // Verify correctness (max absolute difference)
    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabs(hC_cublas[i] - hC_custom[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    // Output results
    std::cout << "cuBLAS SGEMM time: " << ms_cublas << " ms" << std::endl;
    std::cout << "Custom kernel time: " << ms_custom << " ms" << std::endl;
    std::cout << "Speedup (custom / cuBLAS): " << ms_custom / ms_cublas << std::endl;
    std::cout << "Maximum absolute difference between results: " << maxDiff << std::endl;

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC_cublas));
    CUDA_CHECK(cudaFree(dC_custom));
    delete[] hA;
    delete[] hB;
    delete[] hC_cublas;
    delete[] hC_custom;

    return 0;
}
