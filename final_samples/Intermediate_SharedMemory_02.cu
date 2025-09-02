```cpp
/*
Implement tiled matrix multiplication. Each thread block will calculate one square sub-matrix (tile) of C. To do this, all threads in the block must cooperatively load the corresponding tiles of A and B into shared memory first.

Thinking:
1. Use a fixed tile size (e.g., 16 or 32) that matches the GPU architecture's warp size and shared memory capacity.
2. Each thread block processes one tile of the output matrix C. The block dimension is TILE_SIZE x TILE_SIZE.
3. For each tile, threads load sub-tiles of A and B into shared memory:
   - Shared memory arrays As and Bs are declared as __shared__ float arrays of size TILE_SIZE x TILE_SIZE.
   - Each thread loads one element from A and one from B into shared memory.
4. After loading, __syncthreads() ensures all loads are finished before computation.
5. Each thread then performs a partial dot product for the tile: it multiplies corresponding elements of As and Bs across the tile dimension and accumulates into a register (cValue).
6. Repeat the load‑compute cycle for all tiles along the K dimension: loop over k tiles.
7. Finally, each thread writes its computed element to global memory C, with bounds checking to handle matrices whose dimensions are not multiples of TILE_SIZE.
8. On the host, allocate matrices A, B, C (float), initialize A and B (random or simple pattern), copy them to device, launch the kernel, copy back the result, and optionally verify correctness with a CPU reference implementation.
9. Include basic CUDA error checking macros and use cudaDeviceSynchronize() to wait for kernel completion.
10. Keep code self‑contained and free of extraneous output, following the instruction to output only a .cu file.

Note: The tile size chosen here is 32, which is a common choice for modern GPUs. Adjust TILE_SIZE if needed.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__    \
                      << " (" << cudaGetErrorString(err) << ")\n";           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Tile size
constexpr int TILE_SIZE = 32;

// Kernel: tiled matrix multiplication
__global__ void matMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int N, int K) {
    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Identify the row and column of the C element this thread will compute
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float cValue = 0.0f;

    // Loop over all tiles needed to compute C(row, col)
    // Each tile contributes to the dot product across the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load elements of A into shared memory if within bounds
        int aRow = row;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[aRow * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load elements of B into shared memory if within bounds
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < K && bCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + bCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to make sure the tile is loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; ++k) {
            cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// CPU reference implementation for verification
void matMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Helper to compare two matrices
bool compareMatrices(const float* A, const float* B, int size, float epsilon = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

int main() {
    // Matrix dimensions (M x K) * (K x N) = (M x N)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(sizeA);
    float* h_B = (float*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);
    float* h_C_ref = (float*)malloc(sizeC);

    // Initialize input matrices with some values
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>((i * 2) % 100) / 100.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch kernel
    matMulTiled<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());

    // Wait for GPU to finish
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Compute reference result on CPU
    matMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Verify correctness
    bool correct = compareMatrices(h_C, h_C_ref, M * N);
    std::cout << "Matrix multiplication " << (correct ? "PASSED" : "FAILED") << std::endl;

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
```