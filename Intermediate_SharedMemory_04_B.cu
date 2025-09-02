/*
Aim of the program: Use `nsight compute` or `nvprof` to compare the global memory traffic between the two versions. The tiled version should have significantly less traffic.

Thinking:
- To showcase memory traffic differences, I will implement two kernels for matrix‑matrix multiplication of square matrices (size N x N).
- The first kernel (naiveKernel) performs the standard triple loop without any shared memory. Every thread reads its required elements directly from global memory for each multiplication, causing a lot of global memory traffic.
- The second kernel (tiledKernel) uses shared memory tiles of size TILE_SIZE x TILE_SIZE to load sub‑blocks of the input matrices once per block, reuse them for multiple partial products, and therefore reduce the number of global memory accesses.
- Host code allocates device memory, initializes matrices with random data, launches both kernels, and copies the results back to host for a quick correctness check. Timing is optional but can be added for performance comparison.
- The program is written in CUDA C and compiled with nvcc. After compilation, the user can profile each kernel separately using Nsight Compute or nvprof, inspecting the "Global Load" and "Global Store" counters to verify that the tiled version uses far less traffic.
- All code is self‑contained in a single .cu file, as requested.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024                // Matrix dimension (N x N)
#define TILE_SIZE 32          // Tile size for tiled kernel (must divide N)

// Check for CUDA errors
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Naive matrix multiplication kernel (no tiling)
__global__ void naiveKernel(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void tiledKernel(const float *A, const float *B, float *C, int n) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < n && tiledCol < n)
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * n + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < n && tiledCol < n)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * n + tiledCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Utility to fill matrix with random floats
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; ++i) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

// Compare two matrices for correctness
bool compareMatrices(const float *A, const float *B, int n, float epsilon = 1e-3f) {
    for (int i = 0; i < n * n; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main(void) {
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    // Initialize host matrices
    srand(0);
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // ------------------- Naive kernel -------------------
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    naiveKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    printf("Naive kernel elapsed time: %.3f ms\n", ms_naive);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    // ------------------- Tiled kernel -------------------
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    tiledKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_tiled = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms_tiled, start, stop));
    printf("Tiled kernel elapsed time: %.3f ms\n", ms_tiled);

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = compareMatrices(h_C_naive, h_C_tiled, N);
    printf("Result comparison: %s\n", ok ? "PASS" : "FAIL");

    // Clean up
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
