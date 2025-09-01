/*
A common optimization is to load both A and B tiles concurrently, then have a single __syncthreads() before the computation loop. Implement this.
The idea is to perform tiled matrix multiplication where each thread block cooperatively loads a tile of matrix A and a tile of matrix B into shared memory. By loading both tiles in parallel using the same thread indices, we minimize global memory accesses. After both tiles are in shared memory, a single __syncthreads() ensures that all threads have completed loading before we start the inner multiplication loop. This reduces the overhead of synchronization calls, as we only sync once per tile iteration. The kernel then performs the dot product for the current tile and accumulates the result. The host code sets up square matrices of size N, allocates device memory, copies data, launches the kernel, and retrieves the result. Error checking is performed after CUDA API calls. This program demonstrates the optimization and serves as a self‑contained example that can be compiled with nvcc and executed on a CUDA‑capable GPU.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

/* Kernel for tiled matrix multiplication.
 * Loads both A and B tiles into shared memory concurrently,
 * then performs the multiplication for the current tile.
 */
__global__ void MatMulTiled(const float *A, const float *B, float *C, int N)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within the block
    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    // Global row and column of the element to compute
    int row = blockRow * TILE_SIZE + localRow;
    int col = blockCol * TILE_SIZE + localCol;

    float Cvalue = 0.0f;

    // Shared memory for A and B tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over all tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Indices for loading from global memory
        int Arow = row;
        int Acol = t * TILE_SIZE + localCol;

        int Brow = t * TILE_SIZE + localRow;
        int Bcol = col;

        // Load elements into shared memory if within bounds
        if (Arow < N && Acol < N)
            As[localRow][localCol] = A[Arow * N + Acol];
        else
            As[localRow][localCol] = 0.0f;

        if (Brow < N && Bcol < N)
            Bs[localRow][localCol] = B[Brow * N + Bcol];
        else
            Bs[localRow][localCol] = 0.0f;

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            Cvalue += As[localRow][k] * Bs[k][localCol];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

/* Utility function for error checking */
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Host function to initialize matrix with simple pattern */
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
    {
        mat[i] = static_cast<float>(i % 100);
    }
}

int main(int argc, char **argv)
{
    int N = 1024; // Matrix dimension
    if (argc > 1)
        N = atoi(argv[1]);

    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    initMatrix(h_A, N);
    initMatrix(h_B, N);
    memset(h_C, 0, size);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, size), "allocating d_A");
    checkCuda(cudaMalloc((void **)&d_B, size), "allocating d_B");
    checkCuda(cudaMalloc((void **)&d_C, size), "allocating d_C");

    // Copy matrices to device
    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "copying A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "copying B");

    // Configure grid and block dimensions
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    MatMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaGetLastError(), "kernel launch");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "copying C");

    // Simple verification: print a few elements
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[N/2][N/2] = %f\n", h_C[(N/2)*N + N/2]);

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
