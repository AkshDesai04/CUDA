```cuda
/*
Aim: Write a CUDA program that demonstrates why a shared memory version of a computation
(e.g., matrix multiplication) is faster than a naive global memory version, referencing
profiler metrics such as occupancy, memory bandwidth, and instruction throughput.

Thinking:
1. The goal is to show a practical example where shared memory reduces global memory traffic.
2. We will implement two kernels: a naive matrix multiplication kernel that accesses global
   memory directly, and a tiled version that loads sub-matrices into shared memory.
3. The host code will allocate matrices, launch both kernels, time them with CUDA events,
   and print the elapsed times.
4. The comment will explain the expected profiler metrics:
   - The shared memory kernel should have higher occupancy because each thread block
     uses fewer registers (due to the reuse of shared memory tiles).
   - Global memory traffic is reduced: each element of A and B is loaded once per tile
     instead of once per multiply-accumulate operation.
   - Cache utilization is improved, leading to lower global memory latency.
   - Overall, the shared memory version typically shows higher achieved memory
     bandwidth and lower execution time, which is what the profiler metrics will reflect.
5. This program is self‑contained; it can be compiled with `nvcc` and run on any CUDA
   capable GPU.
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <time.h>

// Matrix size
#define N 1024          // N x N matrices
#define TILE_WIDTH 32   // Tile width for shared memory

// Naive matrix multiplication kernel
__global__ void MatMulNaive(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}

// Tiled matrix multiplication using shared memory
__global__ void MatMulShared(const float *A, const float *B, float *C, int width) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        int tiledRow = row;
        int tiledCol = t * TILE_WIDTH + threadIdx.x;
        if (tiledRow < width && tiledCol < width)
            As[threadIdx.y][threadIdx.x] = A[tiledRow * width + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        tiledRow = t * TILE_WIDTH + threadIdx.y;
        tiledCol = col;
        if (tiledRow < width && tiledCol < width)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * width + tiledCol];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = sum;
}

// Helper to check for CUDA errors
void checkCuda(cudaError_t result, const char *msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(result));
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    size_t size = N * N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_shared = (float *)malloc(size);

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc((void **)&d_A, size), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, size), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_C, size), "cudaMalloc d_C");

    checkCuda(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Memcpy h_A->d_A");
    checkCuda(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "Memcpy h_B->d_B");

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    // Timing variables
    cudaEvent_t start, stop;
    float time_naive, time_shared;

    // Naive kernel
    checkCuda(cudaEventCreate(&start), "Event Create start");
    checkCuda(cudaEventCreate(&stop), "Event Create stop");

    checkCuda(cudaEventRecord(start), "Event Record start");
    MatMulNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(stop), "Event Record stop");
    checkCuda(cudaEventSynchronize(stop), "Event Sync stop");
    checkCuda(cudaEventElapsedTime(&time_naive, start, stop), "Event Elapsed naive");

    checkCuda(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost), "Memcpy d_C->h_C_naive");

    // Shared memory kernel
    checkCuda(cudaEventRecord(start), "Event Record start");
    MatMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    checkCuda(cudaEventRecord(stop), "Event Record stop");
    checkCuda(cudaEventSynchronize(stop), "Event Sync stop");
    checkCuda(cudaEventElapsedTime(&time_shared, start, stop), "Event Elapsed shared");

    checkCuda(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost), "Memcpy d_C->h_C_shared");

    printf("Matrix multiplication of size %d x %d\n", N, N);
    printf("Naive kernel time: %f ms\n", time_naive);
    printf("Shared memory kernel time: %f ms\n", time_shared);
    printf("Speedup: %fx\n", time_naive / time_shared);

    // Optional verification (comment out for large N)
    /*
    for (int i = 0; i < N * N; ++i) {
        if (fabs(h_C_naive[i] - h_C_shared[i]) > 1e-4f) {
            printf("Mismatch at %d: naive=%f shared=%f\n", i, h_C_naive[i], h_C_shared[i]);
            break;
        }
    }
    */

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```