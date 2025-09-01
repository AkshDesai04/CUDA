/*
Aim of the program (exactly as requested in the prompt):
Explain in comments how using shared memory reduces the number of global memory reads.

Thinking and explanation:
In a typical CUDA kernel, each thread often accesses the same data from global memory multiple times,
especially when processing data in tiles or performing reductions. Accessing global memory is
slow and costly in terms of bandwidth and latency. Shared memory, on the other hand, is a
user-managed cache that resides on-chip, has very low latency, and can be shared among threads
within the same block.

By loading data from global memory into shared memory once (usually at the beginning of a block)
and then having all threads in the block reuse that shared data, we drastically cut the number of
global memory transactions. For example, consider a simple vector addition where each element of
array A is added to the corresponding element of array B. If each thread reads its own elements
from global memory, we perform 2 global reads per thread. However, if we pre-load blocks of A
and B into shared memory and then let all threads in the block perform the addition using the
shared data, we still need the same number of reads overall, but the access pattern becomes
coalesced, and the shared memory can serve repeated accesses by other threads without additional
global memory traffic. In more complex algorithms like matrix multiplication, each element of
the operand matrices is reused many times; loading them into shared memory once and reusing them
across many threads saves a large number of global reads and improves performance.

This program demonstrates a simple tiled matrix multiplication that uses shared memory to
illustrate the reduction of global memory reads. Each thread block loads a tile of the input
matrices into shared memory once, then performs partial computations using those tiles. The
shared memory buffer is reused by all threads in the block, avoiding repeated global memory
accesses for the same data. The comments inside the kernel explain how and why this reduces
global memory traffic.

The code below is a self-contained .cu file that can be compiled with nvcc and run on a CUDA
capable GPU. It includes a simple CPU host that initializes input matrices, launches the
kernel, and verifies the result against a CPU reference implementation.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Matrix dimensions (must be multiples of TILE_SIZE for simplicity)
#define M 512   // rows of A and C
#define N 512   // columns of B and C
#define K 512   // columns of A and rows of B

// Tile size (block dimension)
#define TILE_SIZE 16

// CUDA kernel for tiled matrix multiplication using shared memory
__global__ void matrixMulShared(const float* A, const float* B, float* C, int m, int n, int k) {
    // Allocate shared memory for tiles of A and B
    __shared__ float Asub[TILE_SIZE][TILE_SIZE];
    __shared__ float Bsub[TILE_SIZE][TILE_SIZE];

    // Identify the row and column of the element in C to compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f; // Accumulator for the computed element

    // Loop over tiles of A and B that are needed to compute this element
    // Each iteration loads one tile of A and one tile of B into shared memory
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Global indices of the elements to load into shared memory
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x; // column within the tile
        int Brow = t * TILE_SIZE + threadIdx.y; // row within the tile
        int Bcol = col;

        // Load elements into shared memory if they are within bounds
        // Otherwise load 0 (so that out-of-bounds accesses don't affect result)
        if (Arow < m && Acol < k)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * k + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        if (Brow < k && Bcol < n)
            Bsub[threadIdx.y][threadIdx.x] = B[Brow * n + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the tile is loaded before computing
        __syncthreads();

        // Compute partial product for this tile
        for (int i = 0; i < TILE_SIZE; ++i) {
            Cvalue += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds
    if (row < m && col < n) {
        C[row * n + col] = Cvalue;
    }
}

// CPU reference implementation for verification
void matrixMulCPU(const float* A, const float* B, float* C, int m, int n, int k) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    // Allocate host memory
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Compute reference result on CPU
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA);
    cudaMalloc((void**)&d_B, sizeB);
    cudaMalloc((void**)&d_C, sizeC);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch kernel with appropriate grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify result
    bool correct = true;
    for (int i = 0; i < M * N; ++i) {
        if (fabs(h_C[i] - h_C_ref[i]) > 1e-3f) {
            correct = false;
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n", i, h_C[i], h_C_ref[i]);
            break;
        }
    }
    printf("Result %s\n", correct ? "correct" : "incorrect");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
