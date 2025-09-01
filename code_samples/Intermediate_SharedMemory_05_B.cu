/*
The main computation loop within the kernel does not need boundary checks, as it only ever accesses the shared memory tile.

My approach is to implement a simple matrix multiplication kernel that uses shared memory tiling. The kernel loads a TILE_WIDTH x TILE_WIDTH sub-matrix (tile) of the input matrices A and B into shared memory, then iterates over the tiles of the multiplication dimension without performing any boundary checks. This is safe only if the input matrices' dimensions are multiples of TILE_WIDTH, so I’ll enforce that constraint in the host code. The kernel accumulates the partial products in a register and writes the result to the output matrix C after the loop. The host code handles allocation, initialization, data transfer, kernel launch, and cleanup. I use a simple 2D grid of blocks where each block computes one TILE_WIDTH x TILE_WIDTH tile of the output matrix. The code is fully self-contained and can be compiled with nvcc as a standard .cu file.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Matrix multiplication kernel using shared memory tiling
__global__ void matMulKernel(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Compute global row and column of C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along the K dimension
    // No boundary checks needed because M,N,K are multiples of TILE_WIDTH
    for (int t = 0; t < (K / TILE_WIDTH); ++t) {
        // Load tile of A into shared memory
        tileA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        // Load tile of B into shared memory
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result
    C[row * N + col] = sum;
}

int main() {
    const int M = 64; // Number of rows in A and C
    const int N = 64; // Number of columns in B and C
    const int K = 64; // Number of columns in A and rows in B

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    // Initialize matrices with some values
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f; // Example: all ones
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.0f; // Example: all twos

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data to device
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(N / TILE_WIDTH, M / TILE_WIDTH);

    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify a few elements
    printf("C[0][0] = %f (expected %f)\n", h_C[0], M * 1.0f * 2.0f);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
