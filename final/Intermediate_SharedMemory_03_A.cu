```cuda
/* 
Aim of the program: 
A common optimization is to load both A and B tiles concurrently, then have a single __syncthreads() before the computation loop. 
This program demonstrates that optimization in a tiled matrix multiplication kernel. 
The kernel loads one element of A and one element of B into shared memory per thread, 
synchronizes once, and then performs the dot product for the current tile. 
The host code sets up two random matrices, launches the kernel, and checks the result for correctness. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cmath>
#include <ctime>

#define TILE_SIZE 16

// Error checking macro
#define CHECK_CUDA(call)                                           \
    do {                                                           \
        cudaError_t err = call;                                    \
        if (err != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    } while (0)

// Tiled matrix multiplication kernel with concurrent tile loading
__global__ void matMulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Thread indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // Row in C
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // Column in C

    float acc = 0.0f;

    // Loop over tiles of input matrices
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Concurrently load tiles of A and B into shared memory
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        if (Arow < M && Acol < K)
            tileA[threadIdx.y][threadIdx.x] = A[Arow * K + Acol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;
        if (Brow < K && Bcol < N)
            tileB[threadIdx.y][threadIdx.x] = B[Brow * N + Bcol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize once after both tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// CPU reference implementation for verification
void matMulCPU(const float *A, const float *B, float *C,
               int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main()
{
    srand(static_cast<unsigned>(time(nullptr)));

    // Matrix dimensions
    const int M = 512; // rows of A, rows of C
    const int K = 512; // columns of A, rows of B
    const int N = 512; // columns of B, columns of C

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // Initialize input matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Compute reference result on CPU
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void **)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify correctness
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double err = fabs(h_C[i] - h_C_ref[i]);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %e\n", max_err);

    // Clean up
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    CHECK_CUDA(cudaFree(d_A)); CHECK_CUDA(cudaFree(d_B)); CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```