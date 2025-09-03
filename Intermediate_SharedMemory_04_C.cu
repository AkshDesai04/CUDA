```cpp
/*
Aim: Plot the GFLOPS of both kernels for a range of matrix sizes (e.g., 256, 512, 1024, 2048).

Thinking:
1. We need two matrix multiplication kernels:
   a. Naive kernel where each thread computes one element of C using a simple triple loop.
   b. Tiled kernel that uses shared memory to reduce global memory traffic.
2. For each matrix size in {256, 512, 1024, 2048}:
   - Allocate host memory for matrices A, B, C.
   - Initialize A and B with random values.
   - Allocate device memory and copy A, B to device.
   - Launch both kernels, timing each with cudaEvent.
   - Compute GFLOPS using the formula: (2 * N^3) / (time_in_seconds * 1e9).
   - Print results in a tabular form: size, naive GFLOPS, tiled GFLOPS.
3. The program will be a single .cu file. It includes error checking, timing, and cleanup.
4. The output is textual; it can be redirected to a file and plotted externally with tools like gnuplot or matplotlib.

The code below follows this plan. It is self‑contained and can be compiled with `nvcc`:
   nvcc -O2 -std=c++11 matmul_gflops.cu -o matmul_gflops
Then run:
   ./matmul_gflops > results.txt
The results can be plotted with your favorite plotting tool.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Check CUDA errors
#define CUDA_CHECK(call)                                                     \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",            \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    }

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    // Tile size
    const int TILE_SIZE = 16;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load elements into shared memory, handling boundary
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < N && tiledCol < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to run kernels and measure performance
void runKernels(int N)
{
    size_t size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size); // for result (unused but needed)

    // Initialize input matrices with random values
    srand(0);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    // Device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Configure grid and block sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive kernel timing
    CUDA_CHECK(cudaEventRecord(start, 0));
    matMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msNaive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msNaive, start, stop));

    // Tiled kernel timing
    CUDA_CHECK(cudaEventRecord(start, 0));
    matMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float msTiled = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msTiled, start, stop));

    // Compute GFLOPS
    double gflopsNaive = (2.0 * N * N * N) / (msNaive / 1000.0) / 1e9;
    double gflopsTiled = (2.0 * N * N * N) / (msTiled / 1000.0) / 1e9;

    printf("%8d %12.2f %12.2f\n", N, gflopsNaive, gflopsTiled);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

int main()
{
    printf("Matrix Size  Naive GFLOPS  Tiled GFLOPS\n");
    int sizes[] = {256, 512, 1024, 2048};
    for (int i = 0; i < sizeof(sizes)/sizeof(sizes[0]); ++i) {
        runKernels(sizes[i]);
    }
    return 0;
}
```