```cuda
/*
Aim of the program:
Compare your tiled kernel's performance against the `cublasSgemm` function from the cuBLAS library. (cuBLAS will be much faster).

Thinking:
The task is to implement a tiled matrix multiplication kernel in CUDA and benchmark it against the highly optimized cuBLAS routine cublasSgemm. To make a fair comparison we must use the same matrix sizes, data, and input layout. The program will:
1. Allocate matrices A (MxK), B (KxN) and result matrices C (MxN) on the host.
2. Initialize A and B with random values.
3. Allocate corresponding device memory for A, B, C, and a separate C for cuBLAS.
4. Copy host data to the device.
5. Launch a tiled kernel that uses shared memory to accumulate partial sums. The kernel is written for column‑major layout so that it can directly be used with cuBLAS (which expects column‑major data). Each tile will be TILE_SIZE x TILE_SIZE (32 is a good choice for modern GPUs).
6. Time the tiled kernel using CUDA events.
7. Create a cublas handle and call cublasSgemm with the same matrices. The leading dimensions are set to the number of rows (M for A and C, K for B). Timing is also measured with CUDA events.
8. Copy the GPU results back to the host and compare the two results for correctness by computing the maximum absolute difference.
9. Compute performance in GFLOPS: 2*M*N*K floating point operations divided by elapsed time.
10. Clean up all resources.

Important implementation details:
- Column‑major indexing: element (row, col) in a matrix with leading dimension lda is stored at index row + col*lda.
- Shared memory tiling: each thread loads one element of A and B into shared memory, then each thread sums over the inner dimension tile.
- The kernel checks bounds when loading to avoid reading out of range for non‑multiples of TILE_SIZE.
- Error‑checking macros for CUDA and cuBLAS calls make debugging easier.
- Random data generation uses rand() / RAND_MAX to fill host matrices.

With this structure we can compile the file with nvcc:
    nvcc -O2 -lcublas matrix_mul_compare.cu -o matrix_mul_compare
and run it. The output will show the time for the tiled kernel, the time for cuBLAS, the GFLOPS for each, and the maximum difference between the two result matrices.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                           \
                    cudaGetErrorString(err), __FILE__, __LINE__);                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

#define CUBLAS_CHECK(call)                                                        \
    do {                                                                          \
        cublasStatus_t status = call;                                            \
        if (status != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n",                        \
                    status, __FILE__, __LINE__);                                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

const int TILE_SIZE = 32;

// Tiled matrix multiplication kernel (column-major layout)
__global__ void sgemm_tiled(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    // Compute row and column index of the element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Accumulate result
    float sum = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Shared memory for a tile of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < numTiles; ++t) {
        // Indices for the element of A and B to load
        int a_row = row;
        int a_col = t * TILE_SIZE + threadIdx.x;

        int b_row = t * TILE_SIZE + threadIdx.y;
        int b_col = col;

        // Load elements into shared memory with bounds check
        if (a_row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[a_row + a_col * M];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (b_row < K && b_col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row + b_col * K];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    if (row < M && col < N) {
        C[row + col * M] = sum;
    }
}

int main()
{
    // Matrix sizes
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);          // Result from tiled kernel
    float *h_C_cublas = (float*)malloc(sizeC);   // Result from cuBLAS

    // Initialize matrices A and B with random values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Device allocations
    float *d_A, *d_B, *d_C, *d_C_cublas;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));
    CUDA_CHECK(cudaMalloc((void**)&d_C_cublas, sizeC));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t startKernel, stopKernel;
    cudaEvent_t startCublas, stopCublas;
    CUDA_CHECK(cudaEventCreate(&startKernel));
    CUDA_CHECK(cudaEventCreate(&stopKernel));
    CUDA_CHECK(cudaEventCreate(&startCublas));
    CUDA_CHECK(cudaEventCreate(&stopCublas));

    // Launch tiled kernel
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE,
                  (M + TILE_SIZE - 1) / TILE_SIZE);

    CUDA_CHECK(cudaEventRecord(startKernel));
    sgemm_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaEventRecord(stopKernel));
    CUDA_CHECK(cudaEventSynchronize(stopKernel));

    // Timing for tiled kernel
    float elapsedKernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedKernel, startKernel, stopKernel));

    // cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDA_CHECK(cudaEventRecord(startCublas));
    // cuBLAS expects column-major matrices
    CUBLAS_CHECK(cublasSgemm(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             M, N, K,
                             &alpha,
                             d_A, M,   // lda = M
                             d_B, K,   // lda = K
                             &beta,
                             d_C_cublas, M)); // lda = M
    CUDA_CHECK(cudaEventRecord(stopCublas));
    CUDA_CHECK(cudaEventSynchronize(stopCublas));

    // Timing for cuBLAS
    float elapsedCublas = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedCublas, startCublas, stopCublas));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, sizeC, cudaMemcpyDeviceToHost));

    // Compute max absolute difference
    float maxDiff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(h_C[i] - h_C_cublas[i]);
        if (diff > maxDiff) maxDiff = diff;
    }

    // Compute GFLOPS: 2*M*N*K operations
    double gflopsKernel = (2.0 * M * N * K) / (elapsedKernel / 1000.0) / 1e9;
    double gflopsCublas = (2.0 * M * N * K) / (elapsedCublas / 1000.0) / 1e9;

    printf("Tiled Kernel Time:   %.3f ms, GFLOPS: %.3f\n", elapsedKernel, gflopsKernel);
    printf("cuBLAS Time:          %.3f ms, GFLOPS: %.3f\n", elapsedCublas, gflopsCublas);
    printf("Max difference:       %.6f\n", maxDiff);

    // Clean up
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(startKernel));
    CUDA_CHECK(cudaEventDestroy(stopKernel));
    CUDA_CHECK(cudaEventDestroy(startCublas));
    CUDA_CHECK(cudaEventDestroy(stopCublas));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_cublas));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cublas);

    return 0;
}
```