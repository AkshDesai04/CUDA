```cuda
/*********************************************************************
Aim of the program:
Modify the tiled matrix multiplication to handle matrix dimensions that
are not perfect multiples of the tile size. Threads that would read past
the matrix boundary should instead load a 0 into the shared memory tile.

Thinking and design:
- Use standard tiled matrix multiplication with shared memory.
- Use a fixed tile width (e.g., 16) defined by TILE_WIDTH.
- For each thread, compute global row and column indices for the result matrix.
- Iterate over the K dimension in tiles. In each iteration:
  - Load A tile into shared memory if the global row/col indices are within bounds,
    otherwise load 0.
  - Load B tile similarly.
  - Synchronize threads before using the tiles.
  - Accumulate the partial product into a local sum.
- After all tiles, write the sum to C if the thread's global indices are within
  the matrix bounds.
- Handle arbitrary matrix dimensions: compute grid and block dimensions
  by rounding up division to cover all elements.
- Add simple error checking for CUDA API calls.
- Provide a small main function that allocates matrices, fills them with
  sample data, runs the kernel, and prints a few result values.
*********************************************************************/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16  // tile size

// CUDA error checking macro
#define CHECK_CUDA(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// Kernel for tiled matrix multiplication with boundary handling
__global__ void MatMulTiled(const float *A, const float *B, float *C,
                            int M, int K, int N)
{
    // Shared memory tiles for A and B
    __shared__ float Asub[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bsub[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column index of C element to compute
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over all tiles of the matrix product
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Global indices for loading A and B
        int Arow = row;
        int Acol = t * TILE_WIDTH + threadIdx.x;
        int Brows = t * TILE_WIDTH + threadIdx.y;
        int Bcol = col;

        // Load element of A into shared memory or 0 if out of bounds
        if (Arow < M && Acol < K)
            Asub[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            Asub[threadIdx.y][threadIdx.x] = 0.0f;

        // Load element of B into shared memory or 0 if out of bounds
        if (Brows < K && Bcol < N)
            Bsub[threadIdx.y][threadIdx.x] = B[Brows * N + Bcol];
        else
            Bsub[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += Asub[threadIdx.y][i] * Bsub[i][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}

// Helper function to initialize matrix with simple pattern
void initMatrix(float *mat, int rows, int cols, int val)
{
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)(val + i % 10);  // arbitrary pattern
    }
}

// Helper function to print a submatrix
void printMatrix(const char *name, const float *mat, int rows, int cols, int maxRows = 8, int maxCols = 8)
{
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < maxRows; ++i) {
        for (int j = 0; j < cols && j < maxCols; ++j) {
            printf("%6.1f ", mat[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    // Example dimensions (not necessarily multiples of TILE_WIDTH)
    int M = 45;   // rows of A and C
    int K = 37;   // columns of A and rows of B
    int N = 52;   // columns of B and C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);

    // Initialize matrices
    initMatrix(h_A, M, K, 1);
    initMatrix(h_B, K, N, 2);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    MatMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Print some results
    printMatrix("A", h_A, M, K);
    printMatrix("B", h_B, K, N);
    printMatrix("C", h_C, M, N);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```