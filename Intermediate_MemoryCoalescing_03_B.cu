```cuda
/*
Aim of the program:
"Look for the gld_transactions and gst_transactions metrics in the profiler. The naive version should have many more read transactions (gld)."

My thinking:
- Provide two versions of a simple compute kernel that involves many global memory reads.
- Naive matrix multiplication where each thread reads entire row of A and entire column of B in a loop.
- This results in many uncoalesced global memory loads for B due to strided access, leading to high gld_transactions.
- Optimized tiled version uses shared memory to bring tiles of A and B into shared memory, ensuring coalesced loads and fewer gld_transactions.
- Host code allocates matrices, initializes them, runs both kernels, checks results, and prints timing.
- Users can run nvprof or Nsight Systems to view gld_transactions and gst_transactions.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024               // Size of the square matrices
#define TILE_SIZE 16         // Tile size for the tiled kernel

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", \
                    #call, __FILE__, __LINE__,                \
                    cudaGetErrorString(err));                 \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

// Naive matrix multiplication kernel (many gld_transactions)
__global__ void MatMulNaive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        // Strided access to B[k*n + col] -> many uncoalesced loads
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Optimized tiled matrix multiplication kernel (fewer gld_transactions)
__global__ void MatMulTiled(const float *A, const float *B, float *C, int n) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory (coalesced)
        if (row < n && (t * TILE_SIZE + threadIdx.x) < n)
            As[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * TILE_SIZE + threadIdx.y) < n && col < n)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Utility to initialize matrix with pseudo-random values
void initMatrix(float *M, int n) {
    for (int i = 0; i < n * n; ++i) {
        M[i] = (float)(rand()) / RAND_MAX;
    }
}

// Utility to compare two matrices
bool compareMatrices(const float *A, const float *B, int n, float eps = 1e-3f) {
    for (int i = 0; i < n * n; ++i) {
        if (fabs(A[i] - B[i]) > eps) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int size = N * N * sizeof(float);

    // Allocate host matrices
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_tiled = (float *)malloc(size);

    // Initialize input matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device matrices
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ---------- Naive kernel ----------
    CUDA_CHECK(cudaEventRecord(start));
    MatMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msNaive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msNaive, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));

    // ---------- Tiled kernel ----------
    CUDA_CHECK(cudaEventRecord(start));
    MatMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTiled = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msTiled, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = compareMatrices(h_C_naive, h_C_tiled, N);
    printf("Verification: %s\n", ok ? "PASSED" : "FAILED");

    // Print timings
    printf("Naive kernel time:   %f ms\n", msNaive);
    printf("Tiled kernel time:   %f ms\n", msTiled);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    return 0;
}
```