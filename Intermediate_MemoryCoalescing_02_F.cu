```cuda
/*
Aim of the program:
Handle non-square matrices and dimensions that are not multiples of the tile size.

Thought process:
- Use tiled matrix multiplication with shared memory to accelerate.
- Ensure code supports matrices of arbitrary size M x K and K x N.
- Use TILE_SIZE=16; blockDim 16x16.
- For each tile, load sub-block of A and B into shared memory, with bounds checks.
- Handle edges: if a thread's global row/col is outside matrix bounds, load zero into shared memory.
- For the final write, check bounds before writing to C.
- Launch grid covering ceil(M/TILE_SIZE) x ceil(N/TILE_SIZE).
- Provide host code to allocate, initialize matrices, copy, launch, copy back, free.
- Use CUDA error checking macro.
- Use command line arguments for matrix sizes, fallback to defaults.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_SIZE 16

// Error checking macro
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                    \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

// Kernel: tiled matrix multiplication
__global__ void MatMulKernel(const float *A, const float *B, float *C,
                             int M, int N, int K) {
    // Compute global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles along K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Shared memory for sub-blocks of A and B
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Load A tile into shared memory
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Utility function to fill a matrix with random floats
void randomize_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // values in [-1, 1]
    }
}

int main(int argc, char *argv[]) {
    int M = 512; // Default dimensions
    int K = 512;
    int N = 512;

    // Override dimensions if provided as command line arguments
    if (argc >= 4) {
        M = atoi(argv[1]);
        K = atoi(argv[2]);
        N = atoi(argv[3]);
    } else {
        printf("Using default dimensions: M=%d, K=%d, N=%d\n", M, K, N);
        printf("Or run: %s <M> <K> <N>\n", argv[0]);
    }

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(size_A);
    float *h_B = (float *)malloc(size_B);
    float *h_C = (float *)malloc(size_C);

    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    srand((unsigned)time(NULL));
    randomize_matrix(h_A, M, K);
    randomize_matrix(h_B, K, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size_C));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 dimBlock(TILE_SIZE, TILE_SIZE);
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // (Optional) Validate a few entries
    bool valid = true;
    for (int i = 0; i < 5 && valid; ++i) {
        int row = rand() % M;
        int col = rand() % N;
        float cpu_val = 0.0f;
        for (int k = 0; k < K; ++k)
            cpu_val += h_A[row * K + k] * h_B[k * N + col];
        float gpu_val = h_C[row * N + col];
        if (fabs(cpu_val - gpu_val) > 1e-3f) {
            printf("Mismatch at (%d,%d): CPU=%f GPU=%f\n",
                   row, col, cpu_val, gpu_val);
            valid = false;
        }
    }
    if (valid) printf("Validation passed.\n");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```