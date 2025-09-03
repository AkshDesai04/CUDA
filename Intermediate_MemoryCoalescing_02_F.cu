```cu
/*
Handle non-square matrices and dimensions that are not multiples of the tile size.

Thinking process:
- The goal is to write a CUDA matrix multiplication that works for any rectangular matrices A (M x K) and B (K x N), producing C (M x N).
- The kernel should use shared memory tiling for performance, but must also correctly handle cases where M, N, or K are not multiples of the chosen tile size (BLOCK_DIM).
- We will set a tile size (e.g., 16) and launch a 2D grid where each thread block computes a BLOCK_DIM x BLOCK_DIM tile of C. Within each block, threads will iterate over tiles of A and B along the K dimension.
- For boundary handling, each thread will check whether its global indices (row, col) are within the output matrix bounds before writing the result. Similarly, when loading tiles from A and B into shared memory, we must guard against reading beyond the matrix bounds.
- The host code will allocate memory for A, B, and C on both host and device, initialize A and B with random values, launch the kernel, copy the result back, and optionally verify against a CPU reference implementation.
- We include a simple error-checking macro and a function to compare GPU and CPU results for correctness.
- All code is self-contained in a single .cu file, with no external dependencies beyond the CUDA runtime.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Block size for tiling (you can adjust this)
#define BLOCK_DIM 16

// Macro for checking CUDA errors following a kernel launch or API call
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// CUDA kernel for tiled matrix multiplication
__global__ void matMulTiled(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    // Shared memory tiles
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

    // Row and column indices of the element this thread will compute
    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + BLOCK_DIM - 1) / BLOCK_DIM; ++t) {
        // Global indices for the elements to load into shared memory
        int aRow = row;
        int aCol = t * BLOCK_DIM + threadIdx.x;
        int bRow = t * BLOCK_DIM + threadIdx.y;
        int bCol = col;

        // Load A into shared memory if within bounds, else 0
        if (aRow < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B into shared memory if within bounds, else 0
        if (bRow < K && bCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < BLOCK_DIM; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < M && col < N)
        C[row * N + col] = sum;
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

// Utility to compare two matrices
bool compareMatrices(const float *A, const float *B, int size)
{
    const float eps = 1e-3f;
    for (int i = 0; i < size; ++i) {
        if (fabsf(A[i] - B[i]) > eps) {
            printf("Mismatch at index %d: GPU = %f, CPU = %f\n",
                   i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char *argv[])
{
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    int M = 1024; // rows of A and C
    int K = 512;  // columns of A / rows of B
    int N = 768;  // columns of B and C

    if (argc == 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    } else {
        printf("Using default dimensions M=%d, K=%d, N=%d\n", M, K, N);
        printf("You can provide custom dimensions: ./matrixMul M K N\n");
    }

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // Initialize matrices with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void **)&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (M + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch the kernel
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Verify correctness
    bool correct = compareMatrices(h_C, h_C_ref, M * N);
    if (correct) {
        printf("Result is correct.\n");
    } else {
        printf("Result is incorrect.\n");
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