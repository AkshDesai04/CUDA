/*
Aim of the program:
Write a naive matrix multiplication kernel (C = A * B). Each thread will be responsible for calculating a single element of the output matrix C. This will involve the thread reading one full row from A and one full column from B from global memory.

Thinking:
1. The kernel will be extremely simple: each thread is assigned a unique (row, column) index in the result matrix C. The thread loops over the shared dimension K, reading the appropriate elements from A and B from global memory, accumulating the dot product into a local sum, and finally writing the result back to C.
2. The matrices are stored in row-major order. Thus, element A[row][k] is accessed at A[row * K + k] and element B[k][col] is accessed at B[k * N + col].
3. The kernel launch configuration will be chosen such that each thread block covers a tile of the result matrix. For simplicity we will use a 2D grid of 2D blocks, each block size being 16x16 (a common choice that works well for most GPUs).
4. In the host code we will:
   - Parse optional command line arguments for matrix dimensions (M, K, N). If not provided, default to small sizes for quick tests.
   - Allocate host matrices A, B, C on the CPU and fill A and B with deterministic values so the result can be verified.
   - Allocate corresponding device buffers and copy host data to device.
   - Launch the kernel and time its execution with CUDA events.
   - Copy the result back to the host, optionally verify against a reference CPU implementation.
   - Free all allocated memory.
5. Error checking will be performed on all CUDA API calls via a simple macro to keep the code readable.
6. The entire program is self-contained in a single .cu file and can be compiled with `nvcc -O2 matrixMulNaive.cu -o matrixMulNaive` and run with optional arguments.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cmath>

// Simple error checking macro
#define CHECK_CUDA(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__       \
                      << " code=" << static_cast<int>(err)                    \
                      << " \"" << cudaGetErrorString(err) << "\"\n";           \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Naive matrix multiplication kernel
// C = A * B
// A is M x K, B is K x N, C is M x N
__global__ void matMulNaive(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index in C
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index in C

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            // A[row][k] -> A[row * K + k]
            // B[k][col] -> B[k * N + col]
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CPU reference implementation for verification
void matMulCPU(const float *A, const float *B, float *C,
               int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Simple command line argument parser
void parseArgs(int argc, char **argv, int &M, int &K, int &N)
{
    // Default sizes
    M = 512;
    K = 512;
    N = 512;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        K = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
    } else {
        std::cout << "Using default matrix sizes M=N=K=512. "
                  << "To override, pass: ./matrixMulNaive M K N\n";
    }
}

int main(int argc, char **argv)
{
    int M, K, N;
    parseArgs(argc, argv, M, K, N);

    size_t sizeA = static_cast<size_t>(M) * K;
    size_t sizeB = static_cast<size_t>(K) * N;
    size_t sizeC = static_cast<size_t>(M) * N;

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA * sizeof(float));
    float *h_B = (float *)malloc(sizeB * sizeof(float));
    float *h_C = (float *)malloc(sizeC * sizeof(float));
    float *h_C_ref = (float *)malloc(sizeC * sizeof(float));

    if (!h_A || !h_B || !h_C || !h_C_ref) {
        std::cerr << "Failed to allocate host matrices\n";
        return EXIT_FAILURE;
    }

    // Initialize matrices A and B with deterministic values
    for (size_t i = 0; i < sizeA; ++i) {
        h_A[i] = static_cast<float>((i % 100) + 1);
    }
    for (size_t i = 0; i < sizeB; ++i) {
        h_B[i] = static_cast<float>((i % 50) + 1);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeA * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_B, sizeB * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void **)&d_C, sizeC * sizeof(float)));

    // Copy host data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 blockSize(16, 16); // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Launch kernel
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    std::cout << "Kernel execution time: " << ms << " ms\n";

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify result against CPU reference
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Compute maximum absolute difference
    double maxErr = 0.0;
    for (size_t i = 0; i < sizeC; ++i) {
        double err = std::abs(h_C[i] - h_C_ref[i]);
        if (err > maxErr) maxErr = err;
    }

    std::cout << "Maximum absolute error between GPU and CPU result: "
              << maxErr << "\n";

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
