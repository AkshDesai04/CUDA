/*
Add boundary checks to the tile loading phase. `if (row < N && col < N) { a_tile[...] = A[...]; } else { a_tile[...] = 0.0f; }`.

Thinking:
The goal of this program is to demonstrate a tiled matrix multiplication on the GPU using CUDA, while ensuring that the tile loading phase correctly handles boundary conditions for matrices that may not be a multiple of the tile size.  
- The kernel loads sub‑tiles of matrices A and B into shared memory.  
- For each element of the tile we check whether the global row/column indices are within the bounds of the input matrices (`row < N && col < N`). If they are, we copy the element; otherwise we write a zero to the shared array.  
- This guarantees that when the tile extends beyond the actual matrix dimensions, the padding values are zeros and do not affect the result.  
- The rest of the kernel performs the standard accumulation of the dot product using the loaded tiles.  
- The host code sets up a simple test where A and B are filled with deterministic values, calls the kernel, and prints a few elements of the result to verify correctness.  
- All memory allocations are checked, and the program cleans up before exiting.  
- TILE_WIDTH is set to 32, but can be adjusted for performance on different GPUs.  
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 32

/* Kernel for tiled matrix multiplication with boundary checks during tile load */
__global__ void MatMulKernel(const float *A, const float *B, float *C, int N)
{
    // Shared memory tiles
    __shared__ float a_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float b_tile[TILE_WIDTH][TILE_WIDTH];

    // Compute row and column index of the element to be computed
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t)
    {
        // Global indices for the element to load into shared memory
        int tiledRow = row;
        int tiledCol = t * TILE_WIDTH + threadIdx.x;

        // Load A tile with boundary checks
        if (tiledRow < N && tiledCol < N)
            a_tile[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            a_tile[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE_WIDTH + threadIdx.y;
        tiledCol = col;

        // Load B tile with boundary checks
        if (tiledRow < N && tiledCol < N)
            b_tile[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            b_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Accumulate the product of the two tiles
        for (int k = 0; k < TILE_WIDTH; ++k)
            value += a_tile[threadIdx.y][k] * b_tile[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result
    if (row < N && col < N)
        C[row * N + col] = value;
}

/* Helper function to initialize a matrix with simple values */
void init_matrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = (float)(i % 256) / 256.0f;
}

/* Main function */
int main()
{
    int N = 1025; // Example size not a multiple of TILE_WIDTH
    size_t bytes = N * N * sizeof(float);

    // Host memory allocation
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (N + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Simple verification: print few elements
    printf("C[0] = %f\n", h_C[0]);
    printf("C[N-1] = %f\n", h_C[N - 1]);
    printf("C[N*N-1] = %f\n", h_C[N * N - 1]);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
