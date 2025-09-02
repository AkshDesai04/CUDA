```c
/*
Aim of the program: Report the execution time speedup of the optimized version over the naive one.

Thinking process:
1. Choose a kernel that can benefit from optimization. Matrix multiplication is a classic example where tiling with shared memory gives a noticeable speedup over a simple naive implementation.
2. Define a matrix size that is large enough to show a measurable difference but small enough to keep the example concise (e.g., 1024x1024).
3. Implement two kernels:
   - Naive kernel: each thread computes one element of C by summing over K with global memory accesses only.
   - Optimized tiled kernel: use shared memory to load sub‑tiles of A and B, reducing global memory traffic and increasing reuse.
4. Use CUDA events (cudaEventRecord, cudaEventElapsedTime) to time the kernels accurately.
5. Allocate host and device memory, initialize matrices with random values.
6. Launch the naive kernel, record time, copy result back to host.
7. Launch the optimized kernel, record time, copy result back to host.
8. Verify that the two result matrices are close (within a small epsilon) to ensure correctness.
9. Compute the speedup factor (naive_time / optimized_time) and print it.
10. Clean up all allocated resources.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define N 1024            // Matrix dimension N x N
#define TILE_SIZE 16      // Tile width for optimized kernel

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Naive matrix multiplication kernel
__global__ void matMulNaive(const float *A, const float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Optimized tiled matrix multiplication kernel using shared memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory
        int tiledRow = row;
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        if (tiledRow < width && tiledCol < width)
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * width + tiledCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        tiledRow = t * TILE_SIZE + threadIdx.y;
        tiledCol = col;
        if (tiledRow < width && tiledCol < width)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRow * width + tiledCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}

// Utility function to fill matrix with random floats
void initMatrix(float *mat, int size) {
    for (int i = 0; i < size; ++i) {
        mat[i] = (float)(rand()) / RAND_MAX;
    }
}

// Verify that two matrices are equal within a tolerance
int verify(const float *ref, const float *test, int size) {
    const float eps = 1e-3f;
    for (int i = 0; i < size; ++i) {
        if (fabs(ref[i] - test[i]) > eps) {
            return 0; // mismatch
        }
    }
    return 1; // all good
}

int main(void) {
    srand(time(NULL));

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C_naive = (float *)malloc(bytes);
    float *h_C_opt = (float *)malloc(bytes);

    initMatrix(h_A, N * N);
    initMatrix(h_B, N * N);

    // Device memory
    float *d_A, *d_B, *d_C_naive, *d_C_opt;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C_naive, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C_opt, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Naive kernel
    CUDA_CHECK(cudaEventRecord(start));
    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    // Optimized kernel
    CUDA_CHECK(cudaEventRecord(start));
    matMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C_opt, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_opt = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_opt, start, stop));

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_opt, d_C_opt, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    if (!verify(h_C_naive, h_C_opt, N * N)) {
        fprintf(stderr, "Verification failed: results differ.\n");
    } else {
        printf("Verification passed: results are identical.\n");
    }

    // Report times and speedup
    printf("Naive kernel time: %.3f ms\n", ms_naive);
    printf("Optimized kernel time: %.3f ms\n", ms_opt);
    printf("Speedup: %.2fx\n", ms_naive / ms_opt);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_naive));
    CUDA_CHECK(cudaFree(d_C_opt));

    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_opt);

    return 0;
}
```