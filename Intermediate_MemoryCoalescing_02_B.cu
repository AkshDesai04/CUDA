/*
Phase 2: `__syncthreads()` to ensure the whole tile is loaded.

This program demonstrates the use of `__syncthreads()` in a tiled matrix multiplication kernel. The kernel loads a tile of the input matrices A and B into shared memory, synchronizes all threads in the block with `__syncthreads()`, and then performs the partial multiplication for the current tile. By synchronizing after loading the tile, we guarantee that all shared memory accesses are valid and complete before any thread proceeds to use the data. The host code sets up a simple example, launches the kernel, and prints the resulting matrix. The code is fully self‑contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16  // tile width

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Tiled matrix multiplication kernel
__global__ void matMulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    // Shared memory tiles
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Row and column index of the element in C computed by this thread
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over tiles of A and B
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load tile of A into shared memory
        int Arow = row;
        int Acol = t * BLOCK_SIZE + threadIdx.x;
        if (Arow < M && Acol < K)
            As[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int Brow = t * BLOCK_SIZE + threadIdx.y;
        int Bcol = col;
        if (Brow < K && Bcol < N)
            Bs[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Ensure all threads have loaded their portion of the tile into shared memory
        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue;
}

// Host function to perform matrix multiplication
void matMul(const float *h_A, const float *h_B, float *h_C,
            int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

// Simple test harness
int main() {
    const int M = 64;  // rows of A and C
    const int K = 128; // columns of A and rows of B
    const int N = 32;  // columns of B and C

    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;

    float *h_A = (float*)malloc(size_A * sizeof(float));
    float *h_B = (float*)malloc(size_B * sizeof(float));
    float *h_C = (float*)malloc(size_C * sizeof(float));

    // Initialize matrices with simple values
    for (int i = 0; i < size_A; ++i) h_A[i] = static_cast<float>(i % 10);
    for (int i = 0; i < size_B; ++i) h_B[i] = static_cast<float>((i * 2) % 7);

    // Perform matrix multiplication on GPU
    matMul(h_A, h_B, h_C, M, N, K);

    // Print a few elements of the result
    printf("C[0][0] = %f\n", h_C[0]);
    printf("C[M-1][N-1] = %f\n", h_C[(M-1)*N + (N-1)]);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
