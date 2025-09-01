```cpp
/*
 Aim of the program: Add boundary checks to the tile loading phase. if (row < N && col < N) { a_tile[...] = A[...]; } else { a_tile[...] = 0.0f; }

 The original tile-based matrix multiplication kernel loads a sub-block (tile) of the input matrix A into shared memory
 without checking if the global indices are within the matrix bounds. This works only for matrices whose dimensions are
 multiples of the tile size, but fails (produces out-of-bounds accesses) otherwise. The task is to modify the loading
 phase so that it safely reads from memory when the global indices are valid and writes zero otherwise.

 Approach:
 1. Define a tile size (e.g., 16). Each block will load a TILE_WIDTH x TILE_WIDTH tile from A.
 2. Use threadIdx.x and threadIdx.y to compute the global row and column of the element we want to load.
 3. Perform the boundary check: if (row < N && col < N) then load A[row*N + col] into shared memory; else
    assign 0.0f. This ensures that out-of-range threads write zeros and never access invalid memory.
 4. Continue with the usual matrix multiplication logic: iterate over tiles of B, accumulate partial sums, synchronize
    after each tile, and write the final result to C.

 The rest of the program is a minimal but complete example: it creates two random N x N matrices,
 copies them to the GPU, runs the kernel, copies the result back, and optionally verifies the result
 against a naive CPU implementation. Proper CUDA error checking is included for robustness.

 The program is self-contained and can be compiled with: nvcc -O2 -o matmul matmul.cu
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16

// CUDA error checking macro
#define CUDA_CHECK(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n",  \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Kernel: tiled matrix multiplication with boundary checks during tile loading
__global__ void matMulKernel(const float *A, const float *B, float *C, int N) {
    // Shared memory tiles for A and B
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    // Compute the row and column index of the element in C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load tile from A into shared memory with boundary check
        int a_row = row;
        int a_col = t * TILE_WIDTH + threadIdx.x;
        if (a_row < N && a_col < N)
            a_tile[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        else
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory with boundary check
        int b_row = t * TILE_WIDTH + threadIdx.y;
        int b_col = col;
        if (b_row < N && b_col < N)
            b_tile[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        else
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < N && col < N)
        C[row * N + col] = sum;
}

// Host function to perform naive matrix multiplication for verification
void cpuMatMul(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int N = 1024; // Default matrix size
    if (argc > 1) {
        N = atoi(argv[1]);
    }

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    // Initialize input matrices with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Optional verification
    cpuMatMul(h_A, h_B, h_C_ref, N);

    // Compare a few elements
    int errors = 0;
    for (int i = 0; i < N && errors < 10; ++i) {
        for (int j = 0; j < N && errors < 10; ++j) {
            float diff = fabs(h_C[i * N + j] - h_C_ref[i * N + j]);
            if (diff > 1e-3f) {
                printf("Mismatch at (%d,%d): GPU=%f CPU=%f diff=%f\n",
                       i, j, h_C[i * N + j], h_C_ref[i * N + j], diff);
                errors++;
            }
        }
    }
    if (errors == 0) {
        printf("All tested elements match.\n");
    } else {
        printf("Total mismatches found: %d\n", errors);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```