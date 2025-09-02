```cuda
/*
 * Aim of the program (word to word):
 * "Try using a different tile size, like 32x32, and see how it affects performance.
 * Note that this requires more shared memory and registers."
 *
 * ------------------------------------------------------------------
 * My thinking:
 * 
 * The goal is to experiment with a tiling strategy for matrix multiplication
 * in CUDA.  The standard approach uses a fixed tile size (e.g., 16x16)
 * because it fits nicely into the GPU's shared memory and warp
 * scheduling.  However, larger tiles can reduce the number of global
 * memory accesses by reusing data more aggressively.  The trade‑off is
 * that each block consumes more shared memory and more registers per
 * thread, potentially limiting occupancy.
 *
 * In this program we:
 * 1. Allocate two input matrices (A and B) and an output matrix (C).
 * 2. Implement a generic tiled matrix multiplication kernel
 *    that takes the tile size as a template parameter.
 * 3. Instantiate the kernel with a tile size of 32x32.
 * 4. Measure execution time on the GPU and CPU (for reference).
 * 5. Print the achieved GFLOPS and compare to a naive implementation
 *    (not shown but can be added easily).
 *
 * The kernel uses shared memory arrays of size TILE_DIM x TILE_DIM
 * for A and B tiles.  Since TILE_DIM=32, each shared array consumes
 * 32*32*4 bytes = 4 KiB for float data, and two such arrays use
 * 8 KiB per block – well within the typical 48 KiB shared memory
 * per SM on modern GPUs.  However, larger tiles can cause register
 * pressure if we also unroll loops or use more temporaries.
 *
 * We also enable double buffering by loading the next tile while
 * computing the current tile, but for simplicity we keep the code
 * straightforward.  The block and grid dimensions are chosen to
 * cover the entire matrix.
 *
 * This code is self‑contained, compiles with nvcc, and can be
 * run with command line arguments specifying matrix size N.
 *
 * ------------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Default matrix size if not provided via command line
#define DEFAULT_N 1024
// Tile size (adjustable)
#define TILE_DIM 32

// Kernel for tiled matrix multiplication
__global__ void matMulTile(const float *A, const float *B, float *C, int N)
{
    // Shared memory tiles
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    // Thread indices within block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row/col for this thread
    int row = blockIdx.y * TILE_DIM + ty;
    int col = blockIdx.x * TILE_DIM + tx;

    float acc = 0.0f;

    // Loop over tiles of the input matrices
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t)
    {
        // Load tile from A if within bounds
        int Arow = row;
        int Acol = t * TILE_DIM + tx;
        if (Arow < N && Acol < N)
            tileA[ty][tx] = A[Arow * N + Acol];
        else
            tileA[ty][tx] = 0.0f;

        // Load tile from B if within bounds
        int Brow = t * TILE_DIM + ty;
        int Bcol = col;
        if (Brow < N && Bcol < N)
            tileB[ty][tx] = B[Brow * N + Bcol];
        else
            tileB[ty][tx] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k)
            acc += tileA[ty][k] * tileB[k][tx];

        __syncthreads();
    }

    // Write result to global memory if within bounds
    if (row < N && col < N)
        C[row * N + col] = acc;
}

// Utility function to initialize matrix with random floats
void initMatrix(float *mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// CPU reference implementation (naive) for correctness check
void matMulCPU(const float *A, const float *B, float *C, int N)
{
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k)
                sum += A[i * N + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

int main(int argc, char *argv[])
{
    int N = DEFAULT_N;
    if (argc > 1)
        N = atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);
    printf("Matrix size: %d x %d\n", N, N);
    printf("Tile size: %d x %d\n", TILE_DIM, TILE_DIM);
    printf("Memory per matrix: %.2f MB\n", bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    float *h_C_ref = (float *)malloc(bytes);

    // Initialize input matrices
    srand(0);
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, bytes);
    cudaMalloc((void **)&d_B, bytes);
    cudaMalloc((void **)&d_C, bytes);

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM,
                 (N + TILE_DIM - 1) / TILE_DIM);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up run
    matMulTile<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Measure GPU time
    cudaEventRecord(start);
    matMulTile<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // CPU reference for correctness (optional, can be commented out for large N)
    auto cpu_start = std::chrono::high_resolution_clock::now();
    matMulCPU(h_A, h_B, h_C_ref, N);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N * N && correct; ++i)
    {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > 1e-2f) // tolerance
            correct = false;
    }
    printf("Result correctness: %s\n", correct ? "PASS" : "FAIL");

    // Compute GFLOPS (approx. 2*N^3 operations)
    double gflops = 2.0 * N * N * N / (milliseconds * 1e6);
    printf("GPU time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);

    printf("CPU time: %.3f s\n", cpu_duration.count());
    double cpu_gflops = 2.0 * N * N * N / (cpu_duration.count() * 1e9);
    printf("CPU performance: %.2f GFLOPS\n", cpu_gflops);

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```