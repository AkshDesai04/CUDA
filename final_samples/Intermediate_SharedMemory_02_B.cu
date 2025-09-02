/*
Aim: Declare the shared memory tiles using `__shared__ float a_tile[TILE_SIZE][TILE_SIZE];` syntax.
The following CUDA program performs a simple matrix multiplication C = A × B using shared memory
tiles to accelerate memory access. The key part of this program demonstrates how to declare and
use shared memory tiles with the specified syntax. The program is written in C with CUDA extensions
and can be compiled with nvcc. The host code sets up small test matrices, copies them to the GPU,
launches the kernel, and copies back the result for verification.

Thinking Process:
1. Define a tile size (`TILE_SIZE`) that is a multiple of the warp size (32) to avoid bank conflicts
   and to ensure good occupancy. For demonstration purposes, 16 is used, but it can be changed.
2. Declare shared memory tiles for sub-matrices of A and B using the required syntax:
   `__shared__ float a_tile[TILE_SIZE][TILE_SIZE];` and similarly for `b_tile`.
3. Use a classic tiled matrix multiplication algorithm:
   - Each thread block loads a TILE_SIZE×TILE_SIZE sub-matrix from A and B into shared memory.
   - Synchronize threads within the block.
   - Compute partial products and accumulate into a register.
   - After processing all tiles, write the result to global memory.
4. Host code:
   - Allocate and initialize host matrices A and B.
   - Allocate device memory and copy data.
   - Determine grid and block dimensions.
   - Launch kernel and measure execution time (optional).
   - Copy result back and free memory.
5. The code is self‑contained and can be compiled with:
   `nvcc -o matrixMul matrixMul.cu`
   Then run: `./matrixMul`

This implementation focuses on demonstrating the shared memory tile declaration and usage, while
providing a complete, runnable example for clarity.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_SIZE 16

// Kernel for tiled matrix multiplication
__global__ void matMulShared(const float *A, const float *B, float *C, int N) {
    // Shared memory tiles for A and B
    __shared__ float a_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float b_tile[TILE_SIZE][TILE_SIZE];

    // Identify thread row and column within the grid
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f; // Accumulator for the dot product

    // Loop over tiles of input matrices
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load elements of A into shared memory
        int aRow = row;
        int aCol = m * TILE_SIZE + threadIdx.x;
        if (aRow < N && aCol < N)
            a_tile[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // Load elements of B into shared memory
        int bRow = m * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < N && bCol < N)
            b_tile[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Ensure tiles are fully loaded

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k)
            value += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];

        __syncthreads(); // Ensure previous computations are done before loading next tile
    }

    // Write the result to global memory
    if (row < N && col < N)
        C[row * N + col] = value;
}

// Helper function to allocate and initialize matrices
void initMatrix(float *mat, int N) {
    for (int i = 0; i < N * N; ++i) {
        mat[i] = (float)(rand() % 10); // Random values 0-9
    }
}

int main(void) {
    const int N = 512; // Matrix dimension (N x N)
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Optional: Verify a few entries
    printf("C[0][0] = %f\n", h_C[0]);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
