/*
Implement this for rectangular matrices as well (e.g. C(M,N) = A(M,K) * B(K,N)).
The goal of this program is to perform matrix multiplication on the GPU for
rectangular matrices using CUDA.  We use a tiled approach with shared memory
to reduce global memory traffic.  The program allocates three matrices
A(MxK), B(KxN), and C(MxN), fills A and B with random values, launches a
CUDA kernel that computes the product, copies the result back to the host,
and verifies the result against a naive CPU implementation.  The code
includes error checking for CUDA API calls, handles arbitrary matrix
dimensions, and prints the timing and correctness of the computation.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define TILE_WIDTH 16

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));         \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

// Kernel for matrix multiplication using shared memory tiling
__global__ void MatMulKernel(float *C, const float *A, const float *B,
                             int M, int K, int N)
{
    // Shared memory for tiles of A and B
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    // Row and column index of the element in C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over sub-matrices of A and B that tile the input matrices
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t)
    {
        // Load tile of A into shared memory
        int aRow = row;
        int aCol = t * TILE_WIDTH + threadIdx.x;
        if (aRow < M && aCol < K)
            sharedA[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        int bRow = t * TILE_WIDTH + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N)
            sharedB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += sharedA[threadIdx.y][i] * sharedB[i][threadIdx.x];

        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Host function to perform matrix multiplication on the GPU
void matrixMulCUDA(float *C, const float *A, const float *B,
                   int M, int K, int N)
{
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    CHECK_CUDA(cudaMalloc((void**)&d_A, size_A));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size_B));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size_C));

    CHECK_CUDA(cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, M, K, N);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
}

// CPU reference implementation for verification
void matrixMulCPU(float *C, const float *A, const float *B,
                  int M, int K, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
    }
}

// Helper to print a matrix (for small matrices)
void printMatrix(const char *name, const float *mat, int rows, int cols)
{
    printf("%s =\n", name);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
            printf("%8.2f ", mat[i * cols + j]);
        printf("\n");
    }
}

// Main function
int main(int argc, char *argv[])
{
    // Example matrix dimensions (can be changed or parsed from arguments)
    int M = 64; // number of rows in A and C
    int K = 128; // number of columns in A and rows in B
    int N = 32; // number of columns in B and C

    // Allocate host memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C_gpu = (float*)malloc(size_C);
    float *h_C_cpu = (float*)malloc(size_C);

    // Initialize A and B with random values
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; ++i)
        h_A[i] = (float)(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = (float)(rand() % 10) / 10.0f;

    // Perform matrix multiplication on GPU
    clock_t start_gpu = clock();
    matrixMulCUDA(h_C_gpu, h_A, h_B, M, K, N);
    clock_t end_gpu = clock();
    double time_gpu = (double)(end_gpu - start_gpu) / CLOCKS_PER_SEC;

    // Perform matrix multiplication on CPU for verification
    clock_t start_cpu = clock();
    matrixMulCPU(h_C_cpu, h_A, h_B, M, K, N);
    clock_t end_cpu = clock();
    double time_cpu = (double)(end_cpu - start_cpu) / CLOCKS_PER_SEC;

    // Verify results
    int correct = 1;
    for (int i = 0; i < M * N; ++i)
    {
        if (fabs(h_C_gpu[i] - h_C_cpu[i]) > 1e-3f)
        {
            correct = 0;
            break;
        }
    }

    printf("GPU time: %f seconds\n", time_gpu);
    printf("CPU time: %f seconds\n", time_cpu);
    printf("Result %s\n", correct ? "matches CPU reference" : "differs from CPU reference");

    // Uncomment below to print small matrices
    // printMatrix("A", h_A, M, K);
    // printMatrix("B", h_B, K, N);
    // printMatrix("C GPU", h_C_gpu, M, N);
    // printMatrix("C CPU", h_C_cpu, M, N);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C_gpu);
    free(h_C_cpu);

    return 0;
}
