```cuda
/*
Aim of the program:
Profile for different tile sizes (e.g., 8x8, 16x16, 32x32) and find the optimal one for your GPU.

Thinking:
The goal is to determine which tile size (block dimensions) gives the best performance for a
GPU kernel that performs matrix multiplication. We will implement a standard shared‑memory
tiled matrix multiplication kernel that can operate with a compile‑time constant tile size.
For each candidate tile size (8, 16, 32) we will launch the kernel, time its execution using
CUDA events, and record the elapsed time. After profiling all tile sizes we will print the
times and report the tile size that achieved the lowest execution time.

Key design decisions:
1. Use a templated kernel (`tileMul<TILE_SIZE>`) so that each tile size gets a distinct
   kernel specialization, avoiding runtime branching.
2. Use a square matrix of size 1024×1024 (or 2048 if the device supports it) which is
   large enough to give meaningful timing results but small enough to fit into typical
   device memory.
3. Allocate host and device matrices in linear (row-major) order.
4. Initialize input matrices with deterministic values (e.g., `A[i] = i`, `B[i] = 2*i`) so
   we can verify correctness if needed.
5. Measure execution time for the kernel only (excluding memory transfers) with
   `cudaEventRecord`/`cudaEventElapsedTime`.
6. After profiling, print each tile size and its elapsed time in milliseconds, and
   highlight the best tile size.
7. Compile with `nvcc -arch=sm_XX` (the user should set the appropriate architecture).

This program is self‑contained; it does not rely on any external libraries beyond the
CUDA runtime and the C++ standard library.
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// Matrix size (NxN)
#define N 1024

// Check CUDA errors
inline void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Tiled matrix multiplication kernel
template <int TILE_SIZE>
__global__ void tileMul(const float *A, const float *B, float *C, int width)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t)
    {
        // Load tiles into shared memory, with bounds checking
        int Arow = row;
        int Acol = t * TILE_SIZE + threadIdx.x;
        if (Arow < width && Acol < width)
            As[threadIdx.y][threadIdx.x] = A[Arow * width + Acol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        int Brow = t * TILE_SIZE + threadIdx.y;
        int Bcol = col;
        if (Brow < width && Bcol < width)
            Bs[threadIdx.y][threadIdx.x] = B[Brow * width + Bcol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k)
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    // Write result
    if (row < width && col < width)
        C[row * width + col] = Cvalue;
}

int main()
{
    const int size = N * N;
    const size_t bytes = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes); // to hold GPU result
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize matrices
    for (int i = 0; i < size; ++i)
    {
        h_A[i] = static_cast<float>(i % 100);
        h_B[i] = static_cast<float>((i * 2) % 100);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, bytes), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, bytes), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, bytes), "cudaMalloc d_C");

    // Copy input matrices to device
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Tile sizes to test
    const int tileSizes[] = {8, 16, 32};
    const int numTests = sizeof(tileSizes) / sizeof(tileSizes[0]);

    float bestTime = FLT_MAX;
    int bestTile = -1;

    // Timing events
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    // Grid dimensions (round up)
    dim3 threadsPerBlock;
    dim3 numBlocks;

    printf("Profiling tile sizes for matrix multiplication (N=%d):\n", N);

    for (int i = 0; i < numTests; ++i)
    {
        int TILE = tileSizes[i];
        threadsPerBlock = dim3(TILE, TILE);
        numBlocks = dim3((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // Warm-up launch (not timed)
        if (TILE == 8)
            tileMul<8><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        else if (TILE == 16)
            tileMul<16><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        else if (TILE == 32)
            tileMul<32><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        checkCuda(cudaGetLastError(), "Kernel launch (warm-up)");

        // Synchronize before timing
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize before timing");

        // Timing
        checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start");
        if (TILE == 8)
            tileMul<8><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        else if (TILE == 16)
            tileMul<16><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        else if (TILE == 32)
            tileMul<32><<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
        checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");

        // Wait for completion
        checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

        float msecTotal = 0;
        checkCuda(cudaEventElapsedTime(&msecTotal, start, stop), "cudaEventElapsedTime");

        printf("Tile %dx%d: %.3f ms\n", TILE, TILE, msecTotal);

        if (msecTotal < bestTime)
        {
            bestTime = msecTotal;
            bestTile = TILE;
        }
    }

    printf("\nBest tile size: %dx%d with %.3f ms\n", bestTile, bestTile, bestTime);

    // Copy result back (optional)
    checkCuda(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_C), "cudaFree d_C");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop),  "cudaEventDestroy stop");

    return EXIT_SUCCESS;
}
```