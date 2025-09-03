```c
/*
Aim of the program:
Compare the performance of the shared memory version to a naive version where each thread reads all its required inputs directly from global memory.

Thinking behind the implementation:
1. Problem choice – Matrix multiplication is a classic example where shared memory tiling provides a clear performance benefit. Each thread computes one element of the result matrix C = A * B.
2. Kernel design – Two kernels are provided:
   - naiveKernel: each thread loops over the entire inner dimension, reading A[i][k] and B[k][j] directly from global memory for each multiplication.
   - sharedKernel: uses a tile-based approach. Each thread block loads a tile of A and B into shared memory, synchronizes, and then performs partial products. This reduces global memory traffic and increases data reuse.
3. Performance measurement – CUDA events are used to time kernel execution. Host timing is omitted to keep the focus on device execution.
4. Validation – After both kernels finish, the results are copied back to host and compared for correctness.
5. Scalability – The matrix dimension N is chosen to be a multiple of the block size to avoid boundary checks and simplify the tiling logic.
6. Error handling – A simple macro CHECK_CUDA is used to catch CUDA API errors.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024            // Matrix dimension (N x N)
#define TILE_SIZE 32      // Tile size for shared memory kernel

// Error checking macro
#define CHECK_CUDA(call)                                            \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",    \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }

// Naive matrix multiplication kernel: each thread computes one element of C
__global__ void naiveKernel(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // column index

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Shared memory tiled matrix multiplication kernel
__global__ void sharedKernel(const float *A, const float *B, float *C, int n) {
    // Shared memory tiles for A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (n + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tile from A
        int aRow = row;
        int aCol = m * TILE_SIZE + threadIdx.x;
        if (aRow < n && aCol < n)
            tileA[threadIdx.y][threadIdx.x] = A[aRow * n + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B
        int bRow = m * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < n && bCol < n)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * n + bCol];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Function to fill matrix with random floats
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; ++i) {
        mat[i] = (float)(rand()) / RAND_MAX;
    }
}

// Function to compare two matrices; returns 1 if they are close, 0 otherwise
int compareMatrices(const float *mat1, const float *mat2, int n) {
    const float eps = 1e-3f;
    for (int i = 0; i < n * n; ++i) {
        if (fabs(mat1[i] - mat2[i]) > eps) {
            return 0;
        }
    }
    return 1;
}

int main() {
    int size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C_naive = (float*)malloc(size);
    float *h_C_shared = (float*)malloc(size);

    // Initialize input matrices
    srand(1234);
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, size));
    CHECK_CUDA(cudaMalloc((void**)&d_B, size));
    CHECK_CUDA(cudaMalloc((void**)&d_C, size));

    // Copy input matrices to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    float ms_time_naive, ms_time_shared;

    // --- Naive kernel timing ---
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    naiveKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms_time_naive, start, stop));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));

    // Clean up events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // --- Shared memory kernel timing ---
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    sharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&ms_time_shared, start, stop));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C_shared, d_C, size, cudaMemcpyDeviceToHost));

    // Clean up events
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    // Verify correctness
    int correct = compareMatrices(h_C_naive, h_C_shared, N);
    if (!correct) {
        printf("Results differ!\n");
    } else {
        printf("Results match.\n");
    }

    // Print timings
    printf("Naive kernel time: %f ms\n", ms_time_naive);
    printf("Shared kernel time: %f ms\n", ms_time_shared);
    if (ms_time_shared > 0.0f) {
        printf("Speedup: %.2f×\n", ms_time_naive / ms_time_shared);
    }

    // Free resources
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```