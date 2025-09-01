```cuda
/*
Aim of the program (word for word):
Analyze the instruction mix reported by the profiler. 
The tiled version should have a higher ratio of arithmetic instructions to memory instructions.

Thinking:
The instruction mix in CUDA profiling shows the proportion of different instruction types
(e.g., arithmetic, memory load/store, etc.) executed by a kernel.  
To demonstrate a higher arithmetic-to-memory ratio for a tiled version,
we will implement two versions of a simple matrix multiplication (C = A * B):
1. A naive kernel that performs global memory loads for each multiplication
   without any tiling or shared memory usage.
2. A tiled kernel that loads sub‑tiles of A and B into shared memory once
   per thread block, reducing the number of global memory accesses and
   increasing the proportion of arithmetic operations (multiplication and addition).

Both kernels will be called on the same randomly initialized matrices of
moderate to large size (e.g., 1024x1024).  We will verify the numerical
correctness of the tiled result against the naive result.  
By running the program under Nsight Systems or Visual Profiler,
the user can observe that the tiled kernel exhibits a larger ratio of
arithmetic instructions compared to memory instructions, as expected.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

// Error checking macro
#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // Each thread computes one element of C
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // K

    if (row < M && col < K) {
        float val = 0.0f;
        for (int e = 0; e < N; ++e) {
            val += A[row * N + e] * B[e * K + col];
        }
        C[row * K + col] = val;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float* A, const float* B, float* C,
                            int M, int N, int K) {
    // Shared memory tiles
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices
    int row = blockIdx.y * TILE_WIDTH + ty; // M
    int col = blockIdx.x * TILE_WIDTH + tx; // K

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load A tile
        int aRow = row;
        int aCol = t * TILE_WIDTH + tx;
        if (aRow < M && aCol < N)
            tileA[ty][tx] = A[aRow * N + aCol];
        else
            tileA[ty][tx] = 0.0f;

        // Load B tile
        int bRow = t * TILE_WIDTH + ty;
        int bCol = col;
        if (bRow < N && bCol < K)
            tileB[ty][tx] = B[bRow * K + bCol];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }
        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = sum;
    }
}

// Host function to compare matrices
bool compareMatrices(const float* h1, const float* h2, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(h1[i] - h2[i]) > epsilon) {
            printf("Mismatch at index %d: %f vs %f\n", i, h1[i], h2[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions: MxN * NxK = MxK
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;

    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C_naive = (float*)malloc(sizeC);
    float* h_C_tiled = (float*)malloc(sizeC);

    // Initialize matrices with random values
    for (int i = 0; i < M * N; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, sizeA));
    CHECK_CUDA(cudaMalloc((void**)&d_B, sizeB));
    CHECK_CUDA(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((K + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Timing utilities
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch naive kernel
    CHECK_CUDA(cudaEventRecord(start));
    matMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float timeNaive = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeNaive, start, stop));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Launch tiled kernel
    CHECK_CUDA(cudaEventRecord(start));
    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float timeTiled = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeTiled, start, stop));

    // Copy tiled result back
    CHECK_CUDA(cudaMemcpy(h_C_tiled, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Verify correctness
    bool ok = compareMatrices(h_C_naive, h_C_tiled, M * K);
    if (ok) {
        printf("Results match.\n");
    } else {
        printf("Results do NOT match.\n");
    }

    // Report timings
    printf("Naive kernel time:   %f ms\n", timeNaive);
    printf("Tiled  kernel time:  %f ms\n", timeTiled);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```