```cuda
/* 
Aim: Modify the tiled matrix multiplication to handle matrix dimensions that are not perfect multiples of the tile size. Threads that would read past the matrix boundary should instead load a 0 into the shared memory tile.

Thinking: 
- Use a classic tiled matrix multiplication kernel with a fixed tile width (e.g., 16). 
- Each thread block processes a TILE_WIDTH × TILE_WIDTH tile of the result matrix C. 
- For each phase (iteration over tiles along the K dimension) we load a TILE_WIDTH × TILE_WIDTH tile from A and B into shared memory. 
- If the global indices for these tiles go beyond the actual matrix dimensions (M, K, N), we must load zeros into the shared memory instead of accessing out of bounds memory. This is handled with if‑statements when copying from global to shared memory. 
- After loading, threads perform the partial multiplication for that tile and accumulate into a private Cvalue. 
- After the loop over all tiles, the thread writes the final value to C only if its global row and column indices are within bounds. 
- The host code allocates matrices with dimensions that are not multiples of TILE_WIDTH to test the boundary handling. It then launches the kernel, copies the result back, and verifies correctness by a naive CPU implementation. 
- The kernel and host code use 32‑bit floating point numbers. 
- All allocations and deallocations are checked for errors. 
- The code is self‑contained in a single .cu file. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Tiled matrix multiplication kernel with boundary handling
__global__ void MatMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int K, int N)
{
    // Shared memory tiles for A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Global row and column for this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float Cvalue = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; ++t)
    {
        // Global indices for the elements to load into shared memory
        int colA = t * TILE_WIDTH + threadIdx.x;   // column index in A (k)
        int rowB = t * TILE_WIDTH + threadIdx.y;   // row index in B (k)

        // Load A into shared memory with boundary check
        if (row < M && colA < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + colA];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B into shared memory with boundary check
        if (rowB < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[rowB * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_WIDTH; ++i)
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    // Write the result to global memory with boundary check
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}

// CPU reference implementation for verification
void MatMulCPU(const float* A, const float* B, float* C, int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

// Helper to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    // Example dimensions (not multiples of TILE_WIDTH)
    int M = 45; // rows of A, rows of C
    int K = 30; // columns of A, rows of B
    int N = 27; // columns of B, columns of C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize host matrices with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)rand() / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, sizeA), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, sizeB), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, sizeC), "cudaMalloc d_C");

    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCudaError(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Define grid and block dimensions
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    MatMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");
    checkCudaError(cudaDeviceSynchronize(), "Kernel sync");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // Verify against CPU reference
    MatMulCPU(h_A, h_B, h_C_ref, M, K, N);

    // Compute error norm
    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i)
    {
        double err = fabs(h_C[i] - h_C_ref[i]);
        if (err > maxErr) maxErr = err;
    }
    printf("Maximum absolute error between GPU and CPU: %e\n", maxErr);

    // Clean up
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    checkCudaError(cudaFree(d_A), "cudaFree d_A");
    checkCudaError(cudaFree(d_B), "cudaFree d_B");
    checkCudaError(cudaFree(d_C), "cudaFree d_C");

    return 0;
}
```