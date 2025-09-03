```cuda
/*
 * Aim of the program:
 * Implement tiled matrix multiplication. Each thread block will calculate one square sub-matrix (tile) of C. To do this, all threads in the block must cooperatively load the corresponding tiles of A and B into shared memory first.
 *
 * Thinking process:
 * 1. Use a tile-based approach where each block computes a TILE_DIM x TILE_DIM sub-matrix of the result C.
 * 2. Each thread in the block will load one element from A and one from B into shared memory for the current tile.
 * 3. Synchronize the block after loading to ensure all shared memory loads are complete.
 * 4. Perform the partial dot product for the current tile in a loop over the tile dimension.
 * 5. Accumulate the partial results in a register, then after processing all tiles, write the final value to global memory.
 * 6. Handle edge cases where the matrix size N is not an exact multiple of TILE_DIM by checking bounds before loading/storing.
 * 7. In the host code, allocate and initialize matrices, copy them to device, launch the kernel with an appropriate grid and block size, and copy the result back.
 * 8. Include basic error checking for CUDA API calls.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Define tile size (must be <= 32 for shared memory constraints on many GPUs)
#define TILE_DIM 32

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel: tiled matrix multiplication
__global__ void tiledMatMul(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N)
{
    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Thread row and column within the output matrix
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float Cvalue = 0.0f;

    // Loop over all the tiles needed to compute this C tile
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; ++t)
    {
        // Global indices of the elements to load
        int tiledCol = t * TILE_DIM + threadIdx.x;
        int tiledRow = t * TILE_DIM + threadIdx.y;

        // Load tile of A into shared memory
        if (row < N && tiledCol < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + tiledCol];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B into shared memory
        if (tiledRow < N && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to make sure the tiles are loaded
        __syncthreads();

        // Multiply the two tiles together
        for (int i = 0; i < TILE_DIM; ++i)
            Cvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write the result back to global memory
    if (row < N && col < N)
        C[row * N + col] = Cvalue;
}

// Host function to initialize a matrix with random values
void initMatrix(float* mat, int N)
{
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

int main(int argc, char* argv[])
{
    int N = 512; // Default matrix size
    if (argc > 1)
        N = atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // Initialize host matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    // h_C will hold the result

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy host matrices to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 dimBlock(TILE_DIM, TILE_DIM, 1);
    dim3 dimGrid((N + TILE_DIM - 1) / TILE_DIM,
                 (N + TILE_DIM - 1) / TILE_DIM,
                 1);

    // Launch kernel
    tiledMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // (Optional) Verify result or perform further processing
    // For brevity, we skip verification here

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("Tiled matrix multiplication of size %d x %d completed successfully.\n", N, N);
    return 0;
}
```