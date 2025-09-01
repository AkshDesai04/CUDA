```cuda
/*
Aim: Implement the naive kernel using a
2D grid of
2D blocks, where each thread's global (x, y) index corresponds to
the C(x, y) element it computes.

Thinking:
1. Use a straightforward matrix multiplication kernel where each thread
   computes one element of the result matrix C. The thread's global
   coordinates are derived from block and thread indices:
   - row index = blockIdx.y * blockDim.y + threadIdx.y
   - col index = blockIdx.x * blockDim.x + threadIdx.x

2. The kernel will check that the computed row and column indices fall
   within the matrix bounds (M x N). If they do, the thread will
   accumulate the dot product of the corresponding row of A and
   column of B. Since this is the naive implementation, no shared
   memory or tiling is used.

3. In host code:
   - Define matrix dimensions M, K, N.
   - Allocate and initialize host matrices A, B, C.
   - Allocate device memory, copy A and B to device.
   - Choose a 2D block size (e.g., 16x16) and compute grid size
     to cover all elements of C.
   - Launch the kernel.
   - Copy result C back to host.
   - Optionally verify a few elements or print a small matrix.

4. Error checking macros are added for CUDA API calls.

This code demonstrates the basic naive approach and can be used as a
baseline for performance comparisons with optimized tiled kernels.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error in file '" << __FILE__                     \
                      << "' in line " << __LINE__ << ": "                       \
                      << cudaGetErrorString(err) << std::endl;                  \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Naive matrix multiplication kernel: each thread computes one element of C
__global__ void matMulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float*          C,
                            int M, int N, int K) // A: MxK, B: KxN, C: MxN
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Global row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Global column index

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int e = 0; e < K; ++e) {
            sum += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Helper function to initialize a matrix with random values
void initMatrix(float* mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Helper function to print a small matrix
void printMatrix(const float* mat, int rows, int cols, int maxRows = 8, int maxCols = 8)
{
    int r = std::min(rows, maxRows);
    int c = std::min(cols, maxCols);
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << std::setw(6) << mat[i * cols + j] << " ";
        }
        if (c < cols) std::cout << "...";
        std::cout << "\n";
    }
    if (r < rows) std::cout << "...\n";
}

int main()
{
    // Matrix dimensions
    const int M = 512; // Rows of A and C
    const int K = 512; // Columns of A, rows of B
    const int N = 512; // Columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    if (!h_A || !h_B || !h_C) {
        std::cerr << "Failed to allocate host matrices.\n";
        return EXIT_FAILURE;
    }

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);
    // h_C will hold the result

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const dim3 dimBlock(16, 16); // 256 threads per block
    const dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                       (M + dimBlock.y - 1) / dimBlock.y);

    // Launch the naive kernel
    matMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Optional: print a small portion of the result
    std::cout << "Result matrix C (partial):\n";
    printMatrix(h_C, M, N);

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```