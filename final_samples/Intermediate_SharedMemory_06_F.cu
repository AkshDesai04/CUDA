```c
/*
Aim of the program: Compare the performance of the shared memory version to a naive version where each thread reads all its required inputs directly from global memory.

Thinking:
- To demonstrate the benefit of shared memory, matrix multiplication (C = A * B) is a classic example. The naive implementation has each thread compute one element of C by looping over the entire shared dimension and reading A[row][k] and B[k][col] from global memory every time.
- The tiled/shared-memory version partitions the matrices into subtiles that fit into the per-block shared memory. Each block loads a tile of A and a tile of B into shared memory once, then performs partial products, synchronizing between tiles.
- We will benchmark both kernels using CUDA events to measure execution time for a moderate matrix size (e.g., 1024 x 1024). The block size is chosen to match the shared memory layout (16x16).
- The program will allocate host and device memory, initialize input matrices with random floats, copy them to device, run both kernels (several iterations to average), copy back results, and verify that the outputs are close (optional but useful).
- Timing results and speedup factor will be printed. Error checking is performed after CUDA API calls and kernel launches.
- The entire code is self-contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 1024          // Matrix dimension (N x N)
#define BLOCK_SIZE 16   // Block size for tiled kernel
#define ITERATIONS 10   // Number of kernel launches for timing

// Macro for error checking
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled (shared memory) matrix multiplication kernel
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile of A
        int aRow = row;
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (aRow < N && aCol < N)
            tileA[threadIdx.y][threadIdx.x] = A[aRow * N + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B
        int bRow = t * BLOCK_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < N && bCol < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int e = 0; e < BLOCK_SIZE; ++e) {
            sum += tileA[threadIdx.y][e] * tileB[e][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Helper to initialize matrix with random floats
void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)(rand()) / RAND_MAX;
    }
}

// Helper to compare two matrices for approximate equality
int compareMatrices(const float *mat1, const float *mat2, int size, float eps) {
    for (int i = 0; i < size; ++i) {
        if (fabsf(mat1[i] - mat2[i]) > eps) {
            return 0; // Not equal
        }
    }
    return 1; // Equal within tolerance
}

int main() {
    srand((unsigned)time(NULL));

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_tiled = (float *)malloc(bytes);

    initMatrix(h_A, N * N);
    initMatrix(h_B, N * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Timing objects
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive kernel timing
    float timeNaive = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matMulNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        timeNaive += ms;
    }
    timeNaive /= ITERATIONS;

    // Copy result back (optional, for verification)
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, bytes, cudaMemcpyDeviceToHost));

    // Tiled kernel timing
    float timeTiled = 0.0f;
    for (int i = 0; i < ITERATIONS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        matMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        timeTiled += ms;
    }
    timeTiled /= ITERATIONS;

    // Copy tiled result back for verification
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    int correct = compareMatrices(h_C_naive, h_C_tiled, N * N, 1e-4f);
    if (!correct) {
        printf("Result verification FAILED!\n");
    } else {
        printf("Result verification PASSED.\n");
    }

    printf("Naive kernel time:   %.3f ms\n", timeNaive);
    printf("Tiled kernel time:   %.3f ms\n", timeTiled);
    printf("Speedup:             %.2fx\n", timeNaive / timeTiled);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);

    return 0;
}
```